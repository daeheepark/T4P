from typing import List, Set
import datetime
from pathlib import Path
import importlib

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from metrics import MR, minADE, minFDE
from utils.optim import WarmupCosLR

from .model_ttt_peragent import ModelTTT
from model import layers


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        lr2: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        weight_decay2: float = 1e-4,
        # MAE
        decoder_depth=4,
        actor_mask_ratio=0.5,
        lane_mask_ratio=0.5,
        loss_weight: List[float] = [1.0, 1.0],
        forecast_loss_weight: List[float] = [1.0, 1.0, 1.0],
        mae_loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr2 = lr2
        self.weight_decay2 = weight_decay2
        self.save_hyperparameters()
        # self.submission_handler = SubmissionAv2()

        self.loss_weight = loss_weight
        self.forecast_loss_weight = forecast_loss_weight
        self.mae_loss_weight = mae_loss_weight

        self.net = ModelTTT(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            actor_mask_ratio=actor_mask_ratio,
            lane_mask_ratio=lane_mask_ratio,
            history_steps=historical_steps,
            future_steps=future_steps,
        )

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")

        self.historical_steps = historical_steps
        self.future_steps = future_steps


    def forward(self, data):
        out_forecast = self.net.forward_forecast(data)
        out_mae = self.net.forward_mae(data, out_forecast)

        out_forecast.update(out_mae)
        return out_forecast

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob

    def cal_loss(self, out, data):
        # Forecast Los
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        forecast_loss = (
                self.forecast_loss_weight[0] * agent_reg_loss 
                + self.forecast_loss_weight[1] * agent_cls_loss
                + self.forecast_loss_weight[2] * others_reg_loss
        )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = out["x_mae_hat"]
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions"][:,:,:self.historical_steps,:] - data["x_centers"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.historical_steps]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])

        # future pred loss
        y_hat = out["y_mae_hat"]
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y"].view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_mask"][:, :, self.historical_steps:]
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, self.future_steps)
        future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        loss = (
            self.loss_weight[0] * forecast_loss
            + self.loss_weight[1] * mae_loss
            )

        return {
            "loss" : loss,
            "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            "cls_loss": agent_cls_loss,
            "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }

    def cal_loss_fre(self, out, data):
        # Forecast Los
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        forecast_loss = (
                self.forecast_loss_weight[0] * agent_reg_loss 
                + self.forecast_loss_weight[1] * agent_cls_loss
                + self.forecast_loss_weight[2] * others_reg_loss
        )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = out["x_mae_hat"].view(-1, self.historical_steps, 2)
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions"][:,:,:self.historical_steps,:] - data["x_centers"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.historical_steps]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])

        # future pred loss
        y_hat = out["y_mae_hat"].view(-1, self.future_steps, 2)
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y"].view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_mask"][:, :, self.historical_steps:]
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, self.future_steps)
        future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        loss = (
            self.loss_weight[0] * forecast_loss
            + self.loss_weight[1] * mae_loss
            )

        return {
            "loss" : loss,
            "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            "cls_loss": agent_cls_loss,
            "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }
    
    def cal_loss_fre_obs(self, out, data, obs_fut_mask):
        # Forecast Los
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2][obs_fut_mask], y[obs_fut_mask])
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        others_reg_mask = others_reg_mask * obs_fut_mask.unsqueeze(1)
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        forecast_loss = (
                self.forecast_loss_weight[0] * agent_reg_loss 
                + self.forecast_loss_weight[1] * agent_cls_loss
                + self.forecast_loss_weight[2] * others_reg_loss
        )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = out["x_mae_hat"].view(-1, self.historical_steps, 2)
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions"][:,:,:self.historical_steps,:] - data["x_centers"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.historical_steps]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])

        # future pred loss
        y_hat = out["y_mae_hat"]#.view(-1, self.future_steps, 2)
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y"]#.view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_mask"][:, :, self.historical_steps:]
        reg_mask[~future_pred_mask] = False
        # reg_mask = reg_mask.view(-1, self.future_steps)
        reg_mask = reg_mask * obs_fut_mask.unsqueeze(1)
        future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        loss = (
            self.loss_weight[0] * forecast_loss
            + self.loss_weight[1] * mae_loss
            )

        return {
            "loss" : loss,
            "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            "cls_loss": agent_cls_loss,
            "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }
    
    # def cal_loss_fre_obs_effi(self, out, data, obs_fut_mask):
        # Forecast Los
        y_hat = out["y_hat"]
        y = data["y_agent"][:, 0]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2][obs_fut_mask], y[obs_fut_mask])
        # agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        # others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        # others_reg_mask = others_reg_mask * obs_fut_mask.unsqueeze(1)
        # others_reg_loss = F.smooth_l1_loss(
        #     y_hat_others[others_reg_mask], y_others[others_reg_mask]
        # )

        # forecast_loss = (
        #         self.forecast_loss_weight[0] * agent_reg_loss 
        #         + self.forecast_loss_weight[1] * agent_cls_loss
        #         + self.forecast_loss_weight[2] * others_reg_loss
        # )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = out["x_mae_hat"].view(-1, self.historical_steps, 2)
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions_4_mae"][:,:,:self.historical_steps,:] - data["x_centers_4_mae"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_hist_4_mae"]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])

        # future pred loss
        y_hat = out["y_mae_hat"]#.view(-1, self.future_steps, 2)
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y_4_mae"]#.view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_fut_4_mae"]
        reg_mask[~future_pred_mask] = False
        # reg_mask = reg_mask.view(-1, self.future_steps)
        reg_mask = reg_mask * obs_fut_mask.unsqueeze(1)
        future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        # loss = (
        #     self.loss_weight[0] * forecast_loss
        #     + self.loss_weight[1] * mae_loss
        #     )

        return {
            # "loss" : loss,
            # "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            # "cls_loss": agent_cls_loss,
            # "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }
    
    def training_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data["y"][:, 0])

        for k, v in losses.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    # def on_test_start(self) -> None:
    #     save_dir = Path("./submission")
    #     save_dir.mkdir(exist_ok=True)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    #     self.submission_handler = SubmissionAv2(
    #         save_dir=save_dir, filename=f"forecast_mae_{timestamp}"
    #     )

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
    
    def configure_ttt_optimizers(self, conf):
        blacklist = set(conf.blacklist)
        whitelist = set(conf.whitelist)
        assert len(blacklist & whitelist) == 0

        update = set()
        freeze = set()

        blacklist_weight_modules, whitelist_weight_modules = set(), set()
        for module in blacklist:
            blacklist_weight_modules.add(getattr(nn, module))
        for module in whitelist:
            whitelist_weight_modules.add(getattr(nn, module))
        blacklist_weight_modules, whitelist_weight_modules = tuple(blacklist_weight_modules), tuple(whitelist_weight_modules)

        assert len(blacklist) == len(blacklist_weight_modules)
        assert len(whitelist) == len(whitelist_weight_modules)
        
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if isinstance(module, whitelist_weight_modules):
                    update.add(full_param_name)
                elif isinstance(module, blacklist_weight_modules):
                    freeze.add(full_param_name)
                elif 'actor_type_embed' in param_name:
                    freeze.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    if conf.update_param:
                        update.add(full_param_name)
                    else:
                        freeze.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = update & freeze
        union_params = update | freeze
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(update))
                ],
                "weight_decay": self.weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]

    def freeze_layers(self, conf):
        if conf.fr_embedding:
            for param in self.net.pos_embed.parameters():
                param.requires_grad = False
            for param in self.net.decoder_pos_embed.parameters():
                param.requires_grad = False
            self.net.actor_type_embed.requires_grad = False
            self.net.lane_type_embed.requires_grad = False
            self.net.history_mask_token.requires_grad = False
            self.net.future_mask_token.requires_grad = False
            self.net.lane_mask_token.requires_grad = False

        if conf.fr_first_layer:
            for param in self.net.hist_embed.parameters():
                param.requires_grad = False
            for param in self.net.future_embed.parameters():
                param.requires_grad = False
            for param in self.net.lane_embed.parameters():
                param.requires_grad = False
            
        if conf.fr_enc_layer:
            for param in self.net.blocks.parameters():
                param.requires_grad = False
            for param in self.net.norm.parameters():
                param.requires_grad = False

        if conf.fr_dec_layer:
            for param in self.net.decoder_embed.parameters():
                param.requires_grad = False
            for param in self.net.decoder_blocks.parameters():
                param.requires_grad = False
            for param in self.net.decoder_norm.parameters():
                param.requires_grad = False

        if conf.fr_last_fore:
            for param in self.net.decoder.parameters():
                param.requires_grad = False
            for param in self.net.dense_predictor.parameters():
                param.requires_grad = False

        if conf.fr_last_mae:
            for param in self.net.lane_pred.parameters():
                param.requires_grad = False
            for param in self.net.history_pred.parameters():
                param.requires_grad = False
            for param in self.net.future_pred.parameters():
                param.requires_grad = False
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder
from .layers.transformer_blocks import Block


class ModelTTT(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        actor_mask_ratio: float = 0.5,
        lane_mask_ratio: float = 0.5,
        history_steps: int = 50,
        future_steps: int = 60,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.lane_mask_ratio = lane_mask_ratio
        

        self.hist_embed = AgentEmbeddingLayer(4, 32, drop_path_rate=drop_path)
        self.future_embed = AgentEmbeddingLayer(3, 32, drop_path_rate=drop_path)
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(5, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.lane_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.history_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.future_pred = nn.Linear(embed_dim, future_steps * 2)
        self.history_pred = nn.Linear(embed_dim, history_steps * 2)
        self.lane_pred = nn.Linear(embed_dim, 20 * 2)

        self.decoder = MultimodalDecoder(embed_dim, future_steps)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
        )

        self.history_steps = history_steps
        self.future_steps = future_steps

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        nn.init.normal_(self.future_mask_token, std=0.02)
        nn.init.normal_(self.lane_mask_token, std=0.02)
        nn.init.normal_(self.history_mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=True)

    @staticmethod
    def agent_random_masking(
        hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        pred_masks = ~future_padding_mask.all(-1)  # [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int()
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []

        device = hist_tokens.device
        agent_ids = torch.arange(hist_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):
            pred_agent_ids = agent_ids[future_pred_mask]
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep]
            fut_ids_keep = pred_agent_ids[fut_ids_keep]
            fut_keep_ids_list.append(fut_ids_keep)

            hist_keep_mask = torch.zeros_like(agent_ids).bool()
            hist_keep_mask[: num_actors[i]] = True
            hist_keep_mask[fut_ids_keep] = False
            hist_ids_keep = agent_ids[hist_keep_mask]
            hist_keep_ids_list.append(hist_ids_keep)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])
            hist_masked_tokens.append(hist_tokens[i, hist_ids_keep])

            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))
            hist_key_padding_mask.append(torch.zeros(len(hist_ids_keep), device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )

    @staticmethod
    def lane_random_masking(x, future_mask_ratio, key_padding_mask):
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)
            x_masked.append(x[i, ids_keep])
            new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))

        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list
    
    def forward_forecast(self, data):
        hist_padding_mask = data["x_padding_mask"][:,:,:self.history_steps]
        hist_key_padding_mask = data["x_key_padding_mask"]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
        B, M, L, D = lane_feat.shape
        lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, self.history_steps-1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][...,0].long()]
        # lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += self.lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed

        x_encoder_rtn = x_encoder.clone()
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        x_agent = x_encoder[:, 0]
        y_hat, pi = self.decoder(x_agent)

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, self.future_steps, 2)

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
            "x_encoder" : x_encoder_rtn,
            "lane_normalized": lane_normalized,
        }
    
    def forward_forecast_peragent(self, data):
        hist_padding_mask = data["x_padding_mask"][:,:,:self.history_steps]
        hist_key_padding_mask = data["x_key_padding_mask"]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
        B, M, L, D = lane_feat.shape
        lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, self.history_steps-1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        # actor_type_embed = self.actor_type_embed[data["x_attr"][...,0].long()]
        # lane_type_embed = self.lane_type_embed.repeat(B, M, 1)

        ###### Per-agent actor_type_embed ######
        assert B == 1
        actor_type_embed = [self.actor_embeds[actor_name] for actor_name in data["actor_names"][0]]
        actor_type_embed = torch.stack(actor_type_embed)
        ########################################

        actor_feat += actor_type_embed
        lane_feat += self.lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed

        x_encoder_rtn = x_encoder.clone()
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        x_agent = x_encoder[:, 0]
        y_hat, pi = self.decoder(x_agent)

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, self.future_steps, 2)

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
            "x_encoder" : x_encoder_rtn,
            "lane_normalized": lane_normalized,
        }
    
    def forward_forecast_peragent_fre(self, data):
        hist_padding_mask = data["x_padding_mask"][:,:,:self.history_steps]
        hist_key_padding_mask = data["x_key_padding_mask"]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
        B, M, L, D = lane_feat.shape
        lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, self.history_steps-1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        # actor_type_embed = self.actor_type_embed[data["x_attr"][...,0].long()]
        # lane_type_embed = self.lane_type_embed.repeat(B, M, 1)

        ###### Per-agent actor_type_embed ######
        assert B == 1
        actor_type_embed = [self.actor_embeds[actor_name] for actor_name in data["actor_names"][0]]
        actor_type_embed = torch.stack(actor_type_embed)
        ########################################

        actor_feat = actor_feat + actor_type_embed
        lane_feat = lane_feat + self.lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed

        x_encoder_rtn = x_encoder.clone()
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        x_agent = x_encoder[:, 0]
        y_hat, pi = self.decoder(x_agent)

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, self.future_steps, 2)

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
            "x_encoder" : x_encoder_rtn,
            "lane_normalized": lane_normalized,
        }

    def forward_mae(self, data, out_forecast=None):

        if out_forecast is not None:
            # Load History and Lane encoding 
            N = data["x"].size(1)
            M = data["lane_positions"].size(1)

            x_encoder = out_forecast["x_encoder"]
            lane_normalized = out_forecast["lane_normalized"]

            assert x_encoder.size(1) == N+M

            hist_feat = x_encoder[:, :N]
            lane_feat = x_encoder[:, -M:]
            
            lane_padding_mask = data["lane_padding_mask"]

            # Future
            future_padding_mask = data["x_padding_mask"][:, :, self.history_steps:]
            future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
            B, N, L, D = future_feat.shape
            future_feat = future_feat.view(B * N, L, D)
            future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
            future_feat = future_feat.view(B, N, future_feat.shape[-1])

            actor_type_embed = self.actor_type_embed[data["x_attr"][..., 0].long()]
            future_feat += actor_type_embed

            x_centers = torch.cat(
                [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
            )
            angles = torch.cat(
                [
                    data["x_angles"][..., self.history_steps-1],
                    data["x_angles"][..., self.history_steps-1],
                    data["lane_angles"],
                ],
                dim=1,
            )
            x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            pos_feat = torch.cat([x_centers, x_angles], dim=-1)

            pos_embed = self.pos_embed(pos_feat)
            future_feat += pos_embed[:, N : N + N]

            
        else:
            # History
            hist_padding_mask = data["x_padding_mask"][:, :, :self.history_steps]
            hist_feat = torch.cat(
                [
                    data["x"],
                    data["x_velocity_diff"][..., None],
                    ~hist_padding_mask[..., None],
                ],
                dim=-1,
            )
            B, N, L, D = hist_feat.shape
            hist_feat = hist_feat.view(B * N, L, D)
            hist_feat = self.hist_embed(hist_feat.permute(0, 2, 1).contiguous())
            hist_feat = hist_feat.view(B, N, hist_feat.shape[-1])

            # Future
            future_padding_mask = data["x_padding_mask"][:, :, self.history_steps:]
            future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
            B, N, L, D = future_feat.shape
            future_feat = future_feat.view(B * N, L, D)
            future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
            future_feat = future_feat.view(B, N, future_feat.shape[-1])

            # Lane
            lane_padding_mask = data["lane_padding_mask"]
            lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
            lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
            B, M, L, D = lane_feat.shape
            lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
            lane_feat = lane_feat.view(B, M, -1)

            ###
            actor_type_embed = self.actor_type_embed[data["x_attr"][..., 0].long()]

            hist_feat += actor_type_embed
            lane_feat += self.lane_type_embed
            future_feat += actor_type_embed

            x_centers = torch.cat(
                [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
            )
            angles = torch.cat(
                [
                    data["x_angles"][..., self.history_steps-1],
                    data["x_angles"][..., self.history_steps-1],
                    data["lane_angles"],
                ],
                dim=1,
            )
            x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            pos_feat = torch.cat([x_centers, x_angles], dim=-1)

            pos_embed = self.pos_embed(pos_feat)
            hist_feat += pos_embed[:, :N]
            lane_feat += pos_embed[:, -M:]
            future_feat += pos_embed[:, N : N + N]
        

        (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            data["num_actors"],
        )

        lane_mask_ratio = self.lane_mask_ratio
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_feat, lane_mask_ratio, data["lane_key_padding_mask"]
        )

        x = torch.cat(
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1],
            fut_masked_tokens.shape[1],
            lane_masked_tokens.shape[1],
        )
        assert x_decoder.shape[1] == Nh + Nf + Nl
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh : Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]

        decoder_hist_token = self.history_mask_token.repeat(B, N, 1)
        hist_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)]
            hist_pred_mask[i, idx] = False

        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False

        decoder_lane_token = self.lane_mask_token.repeat(B, M, 1)
        lane_pred_mask = ~data["lane_key_padding_mask"]
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False

        x_decoder = torch.cat(
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )
        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat)
        decoder_key_padding_mask = torch.cat(
            [
                data["x_key_padding_mask"],
                future_padding_mask.all(-1),
                data["lane_key_padding_mask"],
            ],
            dim=1,
        )

        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        hist_token = x_decoder[:, :N].reshape(-1, self.embed_dim)
        future_token = x_decoder[:, N : 2 * N].reshape(-1, self.embed_dim)
        lane_token = x_decoder[:, -M:]

        lane_pred = self.lane_pred(lane_token).view(B, M, 20, 2)
        x_hat = self.history_pred(hist_token).view(-1, self.history_steps, 2)
        y_hat = self.future_pred(future_token).view(-1, self.future_steps, 2)  # B*N, 120

        return {
            "lane_mae_pred" : lane_pred,
            "lane_pred_mask": lane_pred_mask,
            "x_mae_hat" : x_hat,
            "hist_pred_mask" : hist_pred_mask,
            "y_mae_hat" : y_hat,
            "future_pred_mask" : future_pred_mask,
            }
    
    def forward_mae_fre(self, data, out_forecast=None):

        if out_forecast is not None:
            # Load History and Lane encoding 
            N = data["x"].size(1)
            M = data["lane_positions"].size(1)

            x_encoder = out_forecast["x_encoder"]
            lane_normalized = out_forecast["lane_normalized"]

            assert x_encoder.size(1) == N+M

            hist_feat = x_encoder[:, :N]
            lane_feat = x_encoder[:, -M:]
            
            lane_padding_mask = data["lane_padding_mask"]

            # Future
            future_padding_mask = data["x_padding_mask"][:, :, self.history_steps:]
            future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
            B, N, L, D = future_feat.shape
            future_feat = future_feat.view(B * N, L, D)
            future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
            future_feat = future_feat.view(B, N, future_feat.shape[-1])

            actor_type_embed = self.actor_type_embed[data["x_attr"][..., 0].long()]
            future_feat = future_feat + actor_type_embed

            x_centers = torch.cat(
                [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
            )
            angles = torch.cat(
                [
                    data["x_angles"][..., self.history_steps-1],
                    data["x_angles"][..., self.history_steps-1],
                    data["lane_angles"],
                ],
                dim=1,
            )
            x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            pos_feat = torch.cat([x_centers, x_angles], dim=-1)

            pos_embed = self.pos_embed(pos_feat)
            future_feat = future_feat + pos_embed[:, N : N + N]

            
        else:
            # History
            hist_padding_mask = data["x_padding_mask"][:, :, :self.history_steps]
            hist_feat = torch.cat(
                [
                    data["x"],
                    data["x_velocity_diff"][..., None],
                    ~hist_padding_mask[..., None],
                ],
                dim=-1,
            )
            B, N, L, D = hist_feat.shape
            hist_feat = hist_feat.view(B * N, L, D)
            hist_feat = self.hist_embed(hist_feat.permute(0, 2, 1).contiguous())
            hist_feat = hist_feat.view(B, N, hist_feat.shape[-1])

            # Future
            future_padding_mask = data["x_padding_mask"][:, :, self.history_steps:]
            future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
            B, N, L, D = future_feat.shape
            future_feat = future_feat.view(B * N, L, D)
            future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
            future_feat = future_feat.view(B, N, future_feat.shape[-1])

            # Lane
            lane_padding_mask = data["lane_padding_mask"]
            lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
            lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
            B, M, L, D = lane_feat.shape
            lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
            lane_feat = lane_feat.view(B, M, -1)

            ###
            actor_type_embed = self.actor_type_embed[data["x_attr"][..., 0].long()]

            hist_feat = hist_feat + actor_type_embed
            lane_feat = lane_feat + self.lane_type_embed
            future_feat = future_feat + actor_type_embed

            x_centers = torch.cat(
                [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
            )
            angles = torch.cat(
                [
                    data["x_angles"][..., self.history_steps-1],
                    data["x_angles"][..., self.history_steps-1],
                    data["lane_angles"],
                ],
                dim=1,
            )
            x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            pos_feat = torch.cat([x_centers, x_angles], dim=-1)

            pos_embed = self.pos_embed(pos_feat)
            hist_feat = hist_feat + pos_embed[:, :N]
            lane_feat = lane_feat + pos_embed[:, -M:]
            future_feat = future_feat + pos_embed[:, N : N + N]
        

        (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            data["num_actors"],
        )

        lane_mask_ratio = self.lane_mask_ratio
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_feat, lane_mask_ratio, data["lane_key_padding_mask"]
        )

        x = torch.cat(
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1],
            fut_masked_tokens.shape[1],
            lane_masked_tokens.shape[1],
        )
        assert x_decoder.shape[1] == Nh + Nf + Nl
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh : Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]

        decoder_hist_token = self.history_mask_token.repeat(B, N, 1)
        hist_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)]
            hist_pred_mask[i, idx] = False

        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False

        decoder_lane_token = self.lane_mask_token.repeat(B, M, 1)
        lane_pred_mask = ~data["lane_key_padding_mask"]
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False

        x_decoder = torch.cat(
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )
        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat)
        decoder_key_padding_mask = torch.cat(
            [
                data["x_key_padding_mask"],
                future_padding_mask.all(-1),
                data["lane_key_padding_mask"],
            ],
            dim=1,
        )

        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        hist_token = x_decoder[:, :N].reshape(-1, self.embed_dim)
        future_token = x_decoder[:, N : 2 * N].reshape(-1, self.embed_dim)
        lane_token = x_decoder[:, -M:]

        lane_pred = self.lane_pred(lane_token).view(B, M, 20, 2)
        x_hat = self.history_pred(hist_token).view(-1, self.history_steps, 2)
        y_hat = self.future_pred(future_token).view(-1, self.future_steps, 2)  # B*N, 120

        return {
            "lane_mae_pred" : lane_pred,
            "lane_pred_mask": lane_pred_mask,
            "x_mae_hat" : x_hat.unsqueeze(0),
            "hist_pred_mask" : hist_pred_mask,
            "y_mae_hat" : y_hat.unsqueeze(0),
            "future_pred_mask" : future_pred_mask,
            }
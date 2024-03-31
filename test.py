import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tqdm import tqdm
from copy import deepcopy
import warnings
warnings.filterwarnings(action='ignore')
from collections import defaultdict
import logging

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

import torch.multiprocessing
from torch.nn.utils.rnn import pad_sequence

torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils.debug_utils import backup_modules


MAX_STEP = 10000
do_viz = False

@hydra.main(version_base=None, config_path="conf", config_name="config_test_ttt")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    viz_dir = os.path.join(output_dir, 'viz')
    os.mkdir(viz_dir)
    backup_modules(conf, __file__, output_dir)

    if conf.wandb != "disable":
        logger = WandbLogger(
            project="Forecast-MAE",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    logger2 = logging.getLogger("lightning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'==== Run on device: {device} ====')
    print(logger._save_dir.split('/')[-1])
    print('==============================')

    model = instantiate(conf.model.target)
    model.net.load_from_checkpoint(conf.pretrained_weights)
    model = model.to(device)
    model.eval()
    model.freeze_layers(conf)

    actor_tyme_embed_clone = model.net.actor_type_embed.clone().detach()

    optimizers, schedulers = model.configure_ttt_optimizers(conf)

    datamodule = instantiate(conf.datamodule)
    datamodule.setup()
    testloader = datamodule.test_dataloader()

    fut_masks = torch.ones((conf.model.target.future_steps,conf.model.target.future_steps), device=device, dtype=torch.bool)
    fut_masks = torch.flip(torch.tril(fut_masks), dims=(0,))

    SCENE_ID = None
    rel_scene_ts = 0

    output_mae_ = {}
    test_batch_ = {}

    register_maeoutput = 0
    bi_passed = 0

    for bi, test_batch in enumerate(tqdm(testloader)):
        
        if bi_passed > MAX_STEP:
            break
        test_batch = {k: v.to(device=device) if hasattr(v, 'to') else v for k, v in test_batch.items()}

        
        scenario_id = test_batch['scenario_id']
        scene_ts = test_batch['scene_ts']

        assert datamodule.test_args['bs'] == 1
        scenario_id = scenario_id[0]
        scene_ts = scene_ts[0]

        if scenario_id != SCENE_ID:
            output_mae_ = {}
            test_batch_ = {}
            actor_names_scene = []
            actor_types_scene = []
            
            SCENE_ID = scenario_id
            rel_scene_ts = 0

            if bi_passed == 0:
                actor_names = test_batch["actor_names"][0]
                embeds = actor_tyme_embed_clone[test_batch["x_attr"][0,:,0].long()]
                model.net.actor_embeds = torch.nn.ParameterDict({actor_names[i]: embeds[i] for i in range(len(actor_names))})

                actor_names_scene += actor_names
                actor_types_scene += test_batch["x_attr"][0,:,0].long().cpu().tolist()
            else:
                if conf.sep_ego:
                    ego_embeds = model.net.actor_embeds["ego"].clone()

                if not conf.update_type_embed:
                    pass
                else:
                    actor_unique_types = torch.unique(torch.tensor(actor_types_scene))
                    for a_type in actor_unique_types:
                        actor_type_mask = torch.tensor(actor_types_scene) == a_type
                        actor_name_idx = actor_type_mask.nonzero()[:,0].tolist()
                        type_actor_names = list(map(lambda i: actor_names_scene[i], actor_name_idx))

                        type_actor_embeds = [model.net.actor_embeds[key] for key in type_actor_names]
                        type_actor_embeds = torch.nn.Parameter(torch.stack(type_actor_embeds).mean(0))
                        actor_tyme_embed_clone[a_type] = type_actor_embeds
                        
                del model.net.actor_embeds

                if not conf.sep_ego:
                    actor_names = test_batch["actor_names"][0]
                    embeds = actor_tyme_embed_clone[test_batch["x_attr"][0,:,0].long()]
                    model.net.actor_embeds = torch.nn.ParameterDict({actor_names[i]: embeds[i] for i in range(len(actor_names))})
                else:
                    actor_names = test_batch["actor_names"][0][1:]
                    embeds = actor_tyme_embed_clone[test_batch["x_attr"][0,1:,0].long()]
                    model.net.actor_embeds = torch.nn.ParameterDict({actor_names[i]: embeds[i] for i in range(len(actor_names))})
                    model.net.actor_embeds.update({"ego": ego_embeds})

            optimizer1 = torch.optim.AdamW(model.net.actor_embeds.parameters(), lr=model.lr2, weight_decay=model.weight_decay2)

        else:
            rel_scene_ts += 1

            registered_actors = set(model.net.actor_embeds.keys()) 
            current_actors = set(test_batch['actor_names'][0])

            new_actors = list(current_actors.difference(registered_actors))

            for new_actor in new_actors:
                actor_type = test_batch["x_attr"][0,:,0].long()[test_batch['actor_names'][0].index(new_actor)]

                actor_names_scene.append(new_actor)
                actor_types_scene.append(actor_type.item())

                actor_embed = torch.nn.Parameter(actor_tyme_embed_clone[actor_type])
                model.net.actor_embeds.update({new_actor: actor_embed})
                optimizer1.add_param_group({'params': actor_embed,
                                            'weight_decay': model.weight_decay2})
        
        optimizers[0].zero_grad()
        optimizer1.zero_grad()

        output_forecast = model.net.forward_forecast_peragent_fre(test_batch)
        output_mae = model.net.forward_mae_fre(test_batch, output_forecast)
        output_mae.update(output_forecast)

        if register_maeoutput % conf.ttt_real_freq == 0:
            if len(output_mae_) == 0:
                output_mae_.update(output_mae)
                test_batch_.update(test_batch)
            else:
                for key in output_mae_.keys():
                    if output_mae_[key].size(0) < conf.model.target.future_steps:
                        output_mae_[key] = pad_sequence([*output_mae_[key],output_mae[key][0]], batch_first=True)
                    else:
                        output_mae_[key] = pad_sequence([*output_mae_[key][1:],output_mae[key][0]], batch_first=True)

                for key in test_batch_.keys():
                    if key in ['num_actors', 'num_lanes', 'scenario_id', 'scene_ts', 'track_id', 'origin', 'theta', 'actor_names']:
                        continue
                    else:
                        if test_batch_[key].size(0) < conf.model.target.future_steps:
                            test_batch_[key] = pad_sequence([*test_batch_[key],test_batch[key][0]], batch_first=True)
                        else:
                            test_batch_[key] = pad_sequence([*test_batch_[key][1:],test_batch[key][0]], batch_first=True)


        if bi_passed != 0 and bi % conf.ttt_frequency == 0:
            if len(output_mae_) != 0:
                length, Na, _, _ = output_mae_['y_hat'].shape
                obs_fut_mask = fut_masks[-length:]
                
                losses = model.cal_loss_fre_obs(output_mae_, test_batch_, obs_fut_mask)
                loss = losses['reg_loss'] + losses['mae_loss']
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.gradient_clip_val)

                optimizers[0].step()
                optimizer1.step()

                output_mae_ = {}
                test_batch_ = {}

            register_maeoutput += 1

        metrics = model.val_metrics(output_forecast, test_batch["y"][:, 0])
        logger.log_metrics(metrics, bi)

        bi_passed += 1
    
    epoch_metrics = model.val_metrics.compute()
    epoch_metrics = {str(key)+'_epoch': val for key, val in epoch_metrics.items()}

    logger.log_metrics(epoch_metrics)
    print('+'*30)
    exp_name = logger._save_dir.split('/')[-1]
    print(f'Result of exp {exp_name}')
    print('-'*30)
    for k, v in epoch_metrics.items():
        logger2.info(f'{k}: \t {v.item():.3f}')


if __name__ == "__main__":
    main()
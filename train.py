import os, sys

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils.debug_utils import backup_modules


@hydra.main(version_base=None, config_path="conf", config_name="config_tr_ttt")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir
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

    print('==============================')
    print(logger._save_dir.split('/')[-1])
    print('==============================')

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    model = instantiate(conf.model.target)
    datamodule = instantiate(conf.datamodule)

    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)


if __name__ == "__main__":
    main()

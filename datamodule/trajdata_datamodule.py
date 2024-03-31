from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict
from functools import partial

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from trajdata import ProcessedUnifiedDataset, AgentType
from trajdata.augmentation import Augmentation
from datamodule.fmae_utils import filter_stationary_neighbor, filter_lane_center, to_forecastmae, fmae_collate_fn


class TrajdataDataModule(LightningDataModule):
    def __init__(
        self,
        train_args: Dict,
        val_args: Dict,
        test_args: Dict,
        shuffle: bool = True,
        augmentations: Optional[List[Augmentation]] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
    ):
        super(TrajdataDataModule, self).__init__()
        self.train_args, self.val_args, self.test_args = {}, {}, {}
        for _arg in train_args: self.train_args.update(_arg)
        for _arg in val_args: self.val_args.update(_arg)
        for _arg in test_args: self.test_args.update(_arg)

        self.augmentations = augmentations
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            if 'waymo_train' in self.train_args['data'] and self.train_args['dt'] == 0.1 :
                from datamodule.way_01_dataset import TmpWaymo_01
                self.train_dataset = TmpWaymo_01('/home/user/ssd4tb2/Datasets/.unified_data_cache/waymo_train_forecastmae_0.1_0.9_3.0')
            else:
                self.train_dataset = ProcessedUnifiedDataset(
                    desired_data=self.train_args['data'],
                    model_desc='forecastmae',
                    desired_dt=self.train_args['dt'],
                    history_sec=tuple(self.train_args['history']),
                    future_sec=tuple(self.train_args['future']),
                    only_predict=[AgentType.VEHICLE],
                    agent_interaction_distances=defaultdict(lambda: 30.0),
                    incl_robot_future=False,
                    incl_raster_map=False,
                    incl_vector_map=True,
                    augmentations=None,
                    ego_only=self.train_args['ego_only'],
                    num_workers=self.num_workers,
                    # num_workers=0,
                    verbose=True,
                    data_dirs=self.train_args['data_dir'],
                    transforms=[
                                partial(filter_lane_center, 
                                        extent=[-20,80,-50,50],
                                        max_dist=1,
                                        lseg_len=20),
                                to_forecastmae,
                                ]
                )
            self.val_dataset = ProcessedUnifiedDataset(
                desired_data=self.val_args['data'],
                model_desc='forecastmae',
                desired_dt=self.val_args['dt'],
                history_sec=tuple(self.val_args['history']),
                future_sec=tuple(self.val_args['future']),
                only_predict=[AgentType.VEHICLE],
                agent_interaction_distances=defaultdict(lambda: 30.0),
                incl_robot_future=False,
                incl_raster_map=False,
                incl_vector_map=True,
                augmentations=None,
                ego_only=self.val_args['ego_only'],
                num_workers=self.num_workers,
                # num_workers=0,
                verbose=True,
                data_dirs=self.val_args['data_dir'],
                transforms=[
                            partial(filter_lane_center, 
                                    extent=[-20,80,-50,50],
                                    max_dist=1,
                                    lseg_len=20),
                            to_forecastmae,
                            ]
            )

        else:
            self.test_dataset = ProcessedUnifiedDataset(
                desired_data=self.test_args['data'],
                model_desc='forecastmae',
                desired_dt=self.test_args['dt'],
                history_sec=tuple(self.test_args['history']),
                future_sec=tuple(self.test_args['future']),
                only_predict=[AgentType.VEHICLE],
                agent_interaction_distances=defaultdict(lambda: 30.0),
                incl_robot_future=False,
                incl_raster_map=False,
                incl_vector_map=True,
                augmentations=None,
                ego_only=self.test_args['ego_only'],
                num_workers=0,
                verbose=True,
                data_dirs=self.test_args['data_dir'],
                transforms=[
                            partial(filter_lane_center, 
                                    extent=[-20,80,-50,50],
                                    max_dist=1,
                                    lseg_len=20),
                            to_forecastmae,
                            ]
            )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.train_args['bs'],
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(fmae_collate_fn, batch_augments=self.augmentations),
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_args['bs'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(fmae_collate_fn, batch_augments=None),
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_args['bs'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(fmae_collate_fn, batch_augments=None),
        )
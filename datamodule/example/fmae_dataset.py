from collections import defaultdict
from functools import partial
from tqdm import tqdm
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
sys.path.append('/home/user/ssd4tb2/TTTT_MAE')
import multiprocessing
multiprocessing.set_start_method('fork')

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

from trajdata import AgentBatch, AgentType, ProcessedUnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_agent_batch

from datamodule.fmae_utils import filter_stationary_neighbor, filter_lane_center, to_forecastmae, fmae_collate_fn


def main():

    dataset = ProcessedUnifiedDataset(
        desired_data=[
                      "nusc_mini", 
                    #   "nusc_trainval", 
                    #   "lyft_train",
                    #   "lyft_train_full",
                    #   "lyft_val",
                    #   "lyft_sample",
                    #   "interaction_single",
                    # "waymo_train",
                    #   'waymo_val'
                      ],
        model_desc='forecastmae',
        centric="agent",
        desired_dt=0.5,
        history_sec=(2., 2.),
        future_sec=(6., 6.),
        only_predict=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=False,
        incl_vector_map=True,
        augmentations=None,
        ego_only=True,
        num_workers=20,
        verbose=True,
        # rebuild_cache=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_mini": "datasets/nuScenes",
            # "nusc_trainval": "datasets/nuScenes",
            # "lyft_train" : "datasets/lyft/scenes/train.zarr",
            # "lyft_train_full": "datasets/lyft/scenes/train_full.zarr",
            # "lyft_val" : "datasets/lyft/scenes/validate.zarr",
            # "lyft_sample" : "datasets/lyft/scenes/sample.zarr",
            # "interaction_single" : "datasets/interaction_single",
            # "waymo_train" : "datasets/waymo",
            # "waymo_val" : "datasets/waymo",
        },
        transforms=[
                    # filter_stationary_neighbor, 
                    partial(filter_lane_center, 
                            extent=[-20,80,-50,50],
                            max_dist=1,
                            lseg_len=20),
                    to_forecastmae,
                    ]
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=partial(fmae_collate_fn, batch_augments=dataset.augmentations),
        num_workers=32,
    )

    batch: AgentBatch
    start = time.time()
    for batch in tqdm(dataloader):
        pass
    end = time.time()
    print(f'total time passed = {end-start}, per batch = {(end-start)/len(dataset)*4}')


if __name__ == "__main__":
    main()

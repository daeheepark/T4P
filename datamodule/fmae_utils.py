from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from itertools import compress
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

from trajdata.augmentation import Augmentation, BatchAugmentation
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.batch_element import AgentBatchElement
from trajdata.utils import arr_utils

from trajdata.utils.arr_utils import transform_angles_np, transform_coords_np
from trajdata.utils.state_utils import transform_state_np_2d

import matplotlib.pyplot as plt


def filter_stationary_neighbor(batch_element):
    neighbor_stationary_mask = [not dict_item['is_stationary'] for dict_item in batch_element.neighbor_meta_dicts]

    batch_element.neighbor_names = list(compress(batch_element.neighbor_names, neighbor_stationary_mask))
    batch_element.neighbor_future_extents = list(compress(batch_element.neighbor_future_extents, neighbor_stationary_mask))
    batch_element.neighbor_future_lens_np = batch_element.neighbor_future_lens_np[neighbor_stationary_mask]
    batch_element.neighbor_futures = list(compress(batch_element.neighbor_futures, neighbor_stationary_mask))
    batch_element.neighbor_histories = list(compress(batch_element.neighbor_histories, neighbor_stationary_mask))
    batch_element.neighbor_history_extents = list(compress(batch_element.neighbor_histories, neighbor_stationary_mask))
    batch_element.neighbor_history_lens_np = batch_element.neighbor_history_lens_np[neighbor_stationary_mask]
    batch_element.neighbor_meta_dicts = list(compress(batch_element.neighbor_meta_dicts, neighbor_stationary_mask))
    batch_element.neighbor_types_np = batch_element.neighbor_types_np[neighbor_stationary_mask]
    batch_element.num_neighbors = sum(neighbor_stationary_mask)

    return batch_element

def filter_lane_center(batch_element, extent, max_dist, lseg_len):
    x_min, x_max, y_min, y_max = extent
    radius = max([abs(x_min), abs(x_max), abs(y_min), abs(y_max)])
    
    # world_from_agent_tf = np.linalg.inv(batch_element.agent_from_world_tf)

    lanes = batch_element.vec_map.get_lanes_within(batch_element.curr_agent_state_np.as_format("x,y,z"), radius)
    lane_centers = [lane.center.interpolate(max_dist=max_dist) for lane in lanes]

    lane_positions = []
    lane_paddings = []
    for lane_center in lane_centers:
        # lane_center = lane_center.points[:,:2] - batch_element.cache._obs_frame.position
        # lane_center = lane_center @ np.linalg.inv(batch_element.cache._obs_rot_mat)
        lane_center = transform_coords_np(lane_center.points[:,:2], batch_element.agent_from_world_tf)

        lane_in_mask = (
            (lane_center[:, 0] < x_max)
            & (lane_center[:, 0] > x_min)
            & (lane_center[:, 1] < y_max)
            & (lane_center[:, 1] > y_min)
        )
        lane_center = lane_center[lane_in_mask]
        if len(lane_center) == 0:
            continue

        n_segments = int(np.ceil(len(lane_center) / (lseg_len)))
        n_poses = int(np.ceil(len(lane_center) / n_segments))
        for n in range(n_segments):
            lane_position = np.zeros((lseg_len,2))
            lane_padding = np.ones((lseg_len))

            lane_segment = lane_center[n * n_poses: (n+1) * n_poses]
            lane_len = lane_segment.shape[0]

            lane_position[:lane_len] = lane_segment
            lane_padding[:lane_len] = 0

            lane_positions.append(lane_position)
            lane_paddings.append(lane_padding)

    if len(lane_positions) == 0:
        lane_positions = np.array([]).reshape(-1,lseg_len,2)
        lane_paddings = np.array([]).reshape(-1,lseg_len).astype(np.bool)
    else:
        lane_positions = np.stack(lane_positions)
        lane_paddings = np.stack(lane_paddings).astype(np.bool)

    batch_element.lane_positions = lane_positions
    batch_element.lane_paddings = lane_paddings

    return batch_element

def to_forecastmae(
    batch_element: AgentBatchElement,
) -> Union[AgentBatch, Dict[str, Any]]:
        
    max_history_len = batch_element.agent_history_len
    max_fut_len = batch_element.agent_future_len

    ##########
    origin = batch_element.agent_from_world_tf[:2,2].reshape(1,2)
    cos_agent, sin_agent = torch.tensor(batch_element.agent_from_world_tf[0,0]), torch.tensor(batch_element.agent_from_world_tf[1,0])
    theta = torch.atan2(sin_agent, cos_agent)

    # History
    histories = [batch_element.agent_history_np, *batch_element.neighbor_histories]
    history_lens = [batch_element.agent_history_len, *batch_element.neighbor_history_lens_np]
    padded_histories = arr_utils.pad_sequences(
        histories,
        dtype=torch.float,
        time_dim=-2,
        pad_dir=arr_utils.PadDirection.BEFORE,
        batch_first=True,
        padding_value=np.nan,
    )
    if padded_histories.shape[-2] < max_history_len:
        to_add = max_history_len - padded_histories.shape[-2]
        padded_histories = F.pad(
            padded_histories,
            pad=(0, 0, to_add, 0),
            mode="constant",
            value=np.nan,
        )
    range_tensor = torch.arange(max_history_len).unsqueeze(0).expand(batch_element.num_neighbors+1, -1).flip(-1)
    padding_mask_history = ~(range_tensor < torch.tensor(history_lens).unsqueeze(1))

    # Future
    futures = [batch_element.agent_future_np, *batch_element.neighbor_futures]
    future_lens = [batch_element.agent_future_len, *batch_element.neighbor_future_lens_np]
    paddes_futures = arr_utils.pad_sequences(
        futures,
        dtype=torch.float,
        time_dim=-2,
        pad_dir=arr_utils.PadDirection.AFTER,
        batch_first=True,
        padding_value=np.nan,
    )
    if paddes_futures.shape[-2] < max_fut_len:
        to_add = max_history_len - paddes_futures.shape[-2]
        paddes_futures = F.pad(
            paddes_futures,
            (0, 0, 0, to_add),
            mode="constant",
            value=np.nan,
        )
    range_tensor = torch.arange(max_fut_len).unsqueeze(0).expand(batch_element.num_neighbors+1, -1)
    padding_mask_future = ~(range_tensor < torch.tensor(future_lens).unsqueeze(1))

    scene_id = batch_element.scene_id
    origin = torch.from_numpy(origin).float()
    theta = theta.float()
    states_name = [batch_element.agent_name, *batch_element.neighbor_names]
    states_name = [str(sn) for sn in states_name]
    # states_name = list(bytes(name, 'utf8') for name in states_name)
    states = torch.cat((padded_histories, paddes_futures), dim=-2)
    states_padding = torch.cat((padding_mask_history, padding_mask_future), dim=-1)
    num_nodes = batch_element.num_neighbors+1
    lanes = torch.from_numpy(batch_element.lane_positions).float()
    lanes_padding = torch.from_numpy(batch_element.lane_paddings)
    lanes_num = batch_element.lane_positions.shape[0]
    #########

    x = states[...,:2].clone()
    x_attr = torch.tensor([batch_element.agent_type.value, *batch_element.neighbor_types_np], dtype=torch.float).view(-1,1)
    x_ctrs = states[:,max_history_len-1,:2].clone()
    x_positions = states[:,:, :2].clone()
    x_velocity = torch.norm(states[:,:max_history_len, 3:5], p=2, dim=-1)
    x_velocity_diff = x_velocity[:, :max_history_len].clone()
    # x_heading = states[:,1:,:2] - states[:,:-1,:2]
    # x_heading = torch.atan2(x_heading[:,:,1], x_heading[:,:,0])
    x_heading = arr_utils.angle_wrap(states[:,:,-1])

    agent_headings = x_heading[0][~states_padding[0]][-1]
    if agent_headings >= torch.pi/6:
        command = 1     # LEFT
    elif agent_headings <= -torch.pi/6:
        command = 2     # RIGHT
    else:
        command = 0     # FORWARD

    padding_mask = states_padding

    x[:, max_history_len:] = torch.where(
            (padding_mask[:, max_history_len-1].unsqueeze(-1) | padding_mask[:, max_history_len:]).unsqueeze(-1),
            torch.zeros(num_nodes, max_fut_len, 2),
            x[:, max_history_len:] - x[:, max_history_len-1].unsqueeze(-2),
        )
    x[:, 1:max_history_len] = torch.where(
        (padding_mask[:, :max_history_len-1] | padding_mask[:, 1:max_history_len]).unsqueeze(-1),
        torch.zeros(num_nodes, max_history_len-1, 2),
        x[:, 1:max_history_len] - x[:, :max_history_len-1],
    )
    x[:, 0] = torch.zeros(num_nodes, 2)

    x_velocity_diff[:, 1:max_history_len] = torch.where(
        (padding_mask[:, :max_history_len-1] | padding_mask[:, 1:max_history_len]),
        torch.zeros(num_nodes, max_history_len-1),
        x_velocity_diff[:, 1:max_history_len] - x_velocity_diff[:, :max_history_len-1],
    )
    x_velocity_diff[:, 0] = torch.zeros(num_nodes)

    y = x[:, max_history_len:]

    lane_ctr = lanes[:,0]

    if lanes_num > 1:
        lane_angle = torch.atan2(lanes[:, 1, 1] - lanes[:, 0, 1],
                lanes[:, 1, 0] - lanes[:, 0, 0],)
    else:
        lane_angle = torch.tensor([])

    data = {}

    data["x"] = x[:, :max_history_len]
    data["y"] = y
    data["x_positions"] = x_positions
    data["x_attr"] = x_attr
    data["x_centers"] = x_ctrs
    data["x_angles"] = x_heading
    data["x_velocity"] = x_velocity
    data["x_velocity_diff"] = x_velocity_diff
    data["command"] = command
    data["lane_positions"] = lanes
    data["lane_centers"] = lane_ctr
    data["lane_angles"] = lane_angle

    data["x_padding_mask"] = states_padding
    data["lane_padding_mask"] = lanes_padding
    data["num_actors"] = num_nodes
    data["num_lanes"] = lanes_num

    # data["scenario_id"] = bytes(scene_id, 'utf8')
    data["scenario_id"] = scene_id
    data["scene_ts"] = batch_element.scene_ts
    data["actor_names"] = states_name

    data["origin"] = origin
    data["theta"] = theta
    
    data["env_name"] = batch_element.cache.scene.env_name
    data["map_name"] = batch_element.cache.scene.location

    return data

def fmae_collate_fn(
        batch: List[Dict],
        batch_augments: Optional[List[Augmentation]] = None,
):
    data = {}

    for key in [
        "x",
        "x_attr",
        "x_positions",
        "x_centers",
        "x_angles",
        "x_velocity",
        "x_velocity_diff",
        "lane_positions",
        "lane_centers",
        "lane_angles",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask", "lane_padding_mask"]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=True
        )

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["scene_ts"] = [b["scene_ts"] for b in batch]
    data["track_id"] = [b["actor_names"][0] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"].unsqueeze(0) for b in batch])

    data["actor_names"] = [b["actor_names"] for b in batch]

    return data

def ABE2fmae_collate_fn(
    batch_elems: List[AgentBatchElement],
    batch_augments: Optional[List[BatchAugmentation]] = None,
) -> Union[AgentBatch, Dict[str, Any]]:
        
    batch_size: int = len(batch_elems)

    max_history_len = max([elem.agent_history_len for elem in batch_elems])
    max_fut_len = max([elem.agent_future_len for elem in batch_elems])

    scene_ids = list()
    origins = list()
    thetas = list()

    states = list()
    states_padding = list()
    states_num = list()
    lanes = list()
    lanes_padding = list()
    lanes_num = list()

    elem: AgentBatchElement
    for idx, elem in enumerate(batch_elems):
        origin = elem.agent_from_world_tf[:2,2].reshape(1,2)
        cos_agent, sin_agent = torch.tensor(elem.agent_from_world_tf[0,0]), torch.tensor(elem.agent_from_world_tf[1,0])
        theta = torch.atan2(sin_agent, cos_agent)

        # History
        histories = [elem.agent_history_np, *elem.neighbor_histories]
        history_lens = [elem.agent_history_len, *elem.neighbor_history_lens_np]
        padded_histories = arr_utils.pad_sequences(
            histories,
            dtype=torch.float,
            time_dim=-2,
            pad_dir=arr_utils.PadDirection.BEFORE,
            batch_first=True,
            padding_value=np.nan,
        )
        if padded_histories.shape[-2] < max_history_len:
            to_add = max_history_len - padded_histories.shape[-2]
            padded_histories = F.pad(
                padded_histories,
                pad=(0, 0, to_add, 0),
                mode="constant",
                value=np.nan,
            )
        range_tensor = torch.arange(max_history_len).unsqueeze(0).expand(elem.num_neighbors+1, -1).flip(-1)
        padding_mask_history = ~(range_tensor < torch.tensor(history_lens).unsqueeze(1))

        # Future
        futures = [elem.agent_future_np, *elem.neighbor_futures]
        future_lens = [elem.agent_future_len, *elem.neighbor_future_lens_np]
        paddes_futures = arr_utils.pad_sequences(
            futures,
            dtype=torch.float,
            time_dim=-2,
            pad_dir=arr_utils.PadDirection.AFTER,
            batch_first=True,
            padding_value=np.nan,
        )
        if paddes_futures.shape[-2] < max_fut_len:
            to_add = max_history_len - paddes_futures.shape[-2]
            paddes_futures = F.pad(
                paddes_futures,
                (0, 0, 0, to_add),
                mode="constant",
                value=np.nan,
            )
        range_tensor = torch.arange(max_fut_len).unsqueeze(0).expand(elem.num_neighbors+1, -1)
        padding_mask_future = ~(range_tensor < torch.tensor(future_lens).unsqueeze(1))

        scene_ids.append(elem.scene_id)
        origins.append(torch.from_numpy(origin))
        thetas.append(theta)
        states.append(torch.cat((padded_histories, paddes_futures), dim=-2))
        states_padding.append(torch.cat((padding_mask_history, padding_mask_future), dim=-1))
        states_num.append(elem.num_neighbors+1)
        lanes.append(torch.from_numpy(elem.lane_positions))
        lanes_padding.append(torch.from_numpy(elem.lane_paddings))
        lanes_num.append(elem.lane_positions.shape[0])

    origins = torch.cat(origins)
    thetas = torch.tensor(thetas)
    states = torch.cat(states)
    states_padding = torch.cat(states_padding)
    states_num = torch.tensor(states_num)
    lanes = torch.cat(lanes)
    lanes_padding = torch.cat(lanes_padding)
    lanes_num = torch.tensor(lanes_num)

    num_nodes = states_num.sum()

    x = states[...,:2].clone()
    x_ctrs = states[:,max_history_len-1,:2].clone()
    x_positions = states[:,:max_history_len, :2].clone()
    x_velocity = torch.norm(states[:,:max_history_len, 3:5], p=2, dim=-1)
    x_velocity_diff = x_velocity[:, :max_history_len].clone()

    padding_mask = states_padding

    x[:, max_history_len:] = torch.where(
            (padding_mask[:, max_history_len-1].unsqueeze(-1) | padding_mask[:, max_history_len:]).unsqueeze(-1),
            torch.zeros(num_nodes, max_fut_len, 2),
            x[:, max_history_len:] - x[:, max_history_len-1].unsqueeze(-2),
        )
    x[:, 1:max_history_len] = torch.where(
        (padding_mask[:, :max_history_len-1] | padding_mask[:, 1:max_history_len]).unsqueeze(-1),
        torch.zeros(num_nodes, max_history_len-1, 2),
        x[:, 1:max_history_len] - x[:, :max_history_len-1],
    )
    x[:, 0] = torch.zeros(num_nodes, 2)

    x_velocity_diff[:, 1:max_history_len] = torch.where(
        (padding_mask[:, :max_history_len-1] | padding_mask[:, 1:max_history_len]),
        torch.zeros(num_nodes, max_history_len-1),
        x_velocity_diff[:, 1:max_history_len] - x_velocity_diff[:, :max_history_len-1],
    )
    x_velocity_diff[:, 0] = torch.zeros(num_nodes)

    y = x[:, max_history_len:]

    lane_ctr = lanes[:,0]
    lane_angle = torch.atan2(lanes[:, 1, 1] - lanes[:, 0, 1],
            lanes[:, 1, 0] - lanes[:, 0, 0],)

    data = {}

    data["x"] = x[:, :50]
    data["y"] = y
    data["x_positions"] = x_positions
    data["x_centers"] = x_ctrs
    data["x_velocity"] = x_velocity
    data["x_velocity_diff"] = x_velocity_diff
    data["lane_positions"] = lanes
    data["lane_centers"] = lane_ctr
    data["lane_angles"] = lane_angle

    data["x_key_padding_mask"] = states_padding
    data["lane_key_padding_mask"] = lanes_padding
    data["num_actors"] = states_num
    data["num_lanes"] = lanes_num

    data["scenario_id"] = scene_ids

    data["origin"] = origins
    data["theta"] = thetas

    return data
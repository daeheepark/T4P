# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Visualization utils for Argoverse MF scenarios."""

import math
from pathlib import Path
from typing import Final, Optional, Sequence, Set, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
# from av2.datasets.motion_forecasting.data_schema import (
#     ArgoverseScenario,
#     ObjectType,
#     TrackCategory,
# )
# from av2.map.map_api import ArgoverseStaticMap
# from av2.utils.typing import NDArrayFloat, NDArrayInt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as colors

import torch

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 5
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.5
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_W: Final[float] = 80
_PLOT_BOUNDS_BUFFER_H: Final[float] = 80

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"
# _LANE_SEGMENT_COLOR: Final[str] = "#"

_DEFAULT_ACTOR_COLOR: Final[str] = "#00a4ef"  # "#D3E8EF"
_HISTORY_COLOR: Final[str] = "#d34836"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[int] = 100

# _STATIC_OBJECT_TYPES: Set[ObjectType] = {
#     ObjectType.STATIC,
#     ObjectType.BACKGROUND,
#     ObjectType.CONSTRUCTION,
#     ObjectType.RIDERLESS_BICYCLE,
# }

def visualize_compare(
    # scenario: ArgoverseScenario,
    # scenario_static_map: ArgoverseStaticMap,
    # prediction: np.ndarray = None,
    # timestep: int = 50,
    data_batch: Dict,
    output: Dict,
    noupdate_out,
    tent_out,
    mek_out,
    historical_steps,
    save_path: Path = None,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[0].set_facecolor('xkcd:grey')
    ax[1].set_facecolor('xkcd:grey')

    ####### Best modes #######
    l2_norm = torch.norm(output['y_hat'][..., :2] - data_batch['y'][:,0].unsqueeze(1), dim=-1).sum(dim=-1)
    best_mode = torch.argmin(l2_norm, dim=-1)
    y_hat_best = output['y_hat'][torch.arange(output['y_hat'].shape[0]), best_mode]
    y_hat = torch.cat([y_hat_best.unsqueeze(1), output['y_hat_others']], dim=1).cpu() + data_batch['x_positions'][:,:,historical_steps-1].unsqueeze(-2).cpu()
    
    l2_norm = torch.norm(noupdate_out['y_hat'][..., :2] - data_batch['y'][:,0].unsqueeze(1).cpu(), dim=-1).sum(dim=-1)
    best_mode = torch.argmin(l2_norm, dim=-1)
    y_hat_best = noupdate_out['y_hat'][torch.arange(noupdate_out['y_hat'].shape[0]), best_mode]
    y_hat_noup = torch.cat([y_hat_best.unsqueeze(1), noupdate_out['y_hat_others']], dim=1).cpu() + data_batch['x_positions'][:,:,historical_steps-1].unsqueeze(-2).cpu()
    
    l2_norm = torch.norm(tent_out['y_hat'][..., :2] - data_batch['y'][:,0].unsqueeze(1).cpu(), dim=-1).sum(dim=-1)
    best_mode = torch.argmin(l2_norm, dim=-1)
    y_hat_best = tent_out['y_hat'][torch.arange(tent_out['y_hat'].shape[0]), best_mode]
    y_hat_tent = torch.cat([y_hat_best.unsqueeze(1), tent_out['y_hat_others']], dim=1).cpu() + data_batch['x_positions'][:,:,historical_steps-1].unsqueeze(-2).cpu()
    
    l2_norm = torch.norm(mek_out['y_hat'][..., :2] - data_batch['y'][:,0].unsqueeze(1).cpu(), dim=-1).sum(dim=-1)
    best_mode = torch.argmin(l2_norm, dim=-1)
    y_hat_best = mek_out['y_hat'][torch.arange(mek_out['y_hat'].shape[0]), best_mode]
    y_hat_mek = torch.cat([y_hat_best.unsqueeze(1), mek_out['y_hat_others']], dim=1).cpu() + data_batch['x_positions'][:,:,historical_steps-1].unsqueeze(-2).cpu()

    # y_hat_noup = noupdate_out['y_hat'][:,noupdate_out['pi'].argmax(-1)]
    # y_hat_noup = torch.cat([y_hat_noup, noupdate_out['y_hat_others']], dim=1).cpu() + data_batch['x_positions'][:,:,historical_steps-1].unsqueeze(-2).cpu()
    
    # y_hat_tent = tent_out['y_hat'][:,tent_out['pi'].argmax(-1)]
    # y_hat_tent = torch.cat([y_hat_tent, tent_out['y_hat_others']], dim=1).cpu() + data_batch['x_positions'][:,:,historical_steps-1].unsqueeze(-2).cpu()

    # y_hat_mek = mek_out['y_hat'][:,mek_out['pi'].argmax(-1)]
    # y_hat_mek = torch.cat([y_hat_mek, mek_out['y_hat_others']], dim=1).cpu() + data_batch['x_positions'][:,:,historical_steps-1].unsqueeze(-2).cpu()

    ##########################

    # Plot static map elements and actor tracks
    _plot_static_map_elements(data_batch['lane_positions'], data_batch['lane_padding_mask'], ax=ax[0])
    _plot_static_map_elements(data_batch['lane_positions'], data_batch['lane_padding_mask'], ax=ax[1])

    veh_mask = (data_batch['x_attr']==1)[...,0]
    cur_plot_bounds = _plot_actor_tracks_all(ax[0], data_batch['x'][veh_mask].cpu(), data_batch['x_positions'][veh_mask].cpu(), data_batch['x_angles'][veh_mask].cpu(), data_batch['y'][veh_mask].cpu(), data_batch['x_padding_mask'][veh_mask].cpu(), historical_steps)
    cur_plot_bounds = _plot_actor_tracks_all(ax[1], data_batch['x'][veh_mask].cpu(), data_batch['x_positions'][veh_mask].cpu(), data_batch['x_angles'][veh_mask].cpu(), data_batch['y'][veh_mask].cpu(), data_batch['x_padding_mask'][veh_mask].cpu(), historical_steps)
    plot_bounds = cur_plot_bounds


    # prediction = output['y_hat'][0][:,~data_batch['x_padding_mask'][0,0,historical_steps:]].detach().cpu().numpy()

    veh_mask = veh_mask.cpu()
    padding_mask = data_batch['x_padding_mask'][veh_mask][:, historical_steps:].cpu()

    prediction = y_hat_noup[veh_mask]    
    # for ai in range(prediction.size(0)):
    for ai in range(1):
        prediction_ = prediction[ai][~padding_mask[ai]].unsqueeze(0).detach().numpy()
        _scatter_polylines(
            prediction_[:, :, :],
            ax=ax[0],
            color="mediumblue",
            grad_color=False,
            alpha=0.8,
            scale=0.5,
            linewidth=2,
            zorder=1000,
            headwidth=1,
            headlength=1,
        )

    prediction = y_hat_tent[veh_mask]    
    # for ai in range(prediction.size(0)):
    for ai in range(1):
        prediction_ = prediction[ai][~padding_mask[ai]].unsqueeze(0).detach().numpy()
        _scatter_polylines(
            prediction_[:, :, :],
            ax=ax[1],
            color="orange",
            grad_color=False,
            alpha=0.8,
            scale=0.5,
            linewidth=2,
            zorder=999,
            headwidth=1,
            headlength=1,
        )

    prediction = y_hat_mek[veh_mask]    
    # for ai in range(prediction.size(0)):
    for ai in range(1):
        prediction_ = prediction[ai][~padding_mask[ai]].unsqueeze(0).detach().numpy()
        _scatter_polylines(
            prediction_[:, :, :],
            ax=ax[1],
            color="lime",
            grad_color=False,
            alpha=0.8,
            scale=0.5,
            linewidth=2,
            zorder=998,
            headwidth=1,
            headlength=1,
        )

    prediction = y_hat[veh_mask]    
    # for ai in range(prediction.size(0)):
    for ai in range(1):
        prediction_ = prediction[ai][~padding_mask[ai]].unsqueeze(0).detach().numpy()
        _scatter_polylines(
            prediction_[:, :, :],
            ax=ax[1],
            color="mediumblue",
            grad_color=False,
            alpha=0.8,
            scale=0.5,
            linewidth=2,
            zorder=1000,
            headwidth=1,
            headlength=1,
        )

    

    ax[0].set_aspect("equal")
    ax[0].set_xlim(
        plot_bounds[0] - _PLOT_BOUNDS_BUFFER_W, plot_bounds[0] + _PLOT_BOUNDS_BUFFER_W
    )
    ax[0].set_ylim(
        plot_bounds[1] - _PLOT_BOUNDS_BUFFER_H, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_H
    )
    
    ax[1].set_aspect("equal")
    ax[1].set_xlim(
        plot_bounds[0] - _PLOT_BOUNDS_BUFFER_W, plot_bounds[0] + _PLOT_BOUNDS_BUFFER_W
    )
    ax[1].set_ylim(
        plot_bounds[1] - _PLOT_BOUNDS_BUFFER_H, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_H
    )

    plt.tight_layout()
    plt.title(save_path.split('/')[-1].split('.')[0])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)
    else:
        plt.show()

    plt.close('all')

def visualize_fore(
    # scenario: ArgoverseScenario,
    # scenario_static_map: ArgoverseStaticMap,
    # prediction: np.ndarray = None,
    # timestep: int = 50,
    data_batch: Dict,
    output: Dict,
    historical_steps,
    save_path: Path = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # ax.set_aspect('equal')

    ax.set_facecolor('xkcd:grey')
    # ax.set_facecolor((1.0, 0.47, 0.42))

    ####### Best modes #######
    l2_norm = torch.norm(output['y_hat'][..., :2] - data_batch['y'][:,0].unsqueeze(1), dim=-1).sum(dim=-1)
    best_mode = torch.argmin(l2_norm, dim=-1)
    y_hat_best = output['y_hat'][torch.arange(output['y_hat'].shape[0]), best_mode]
    y_hat = torch.cat([y_hat_best.unsqueeze(1), output['y_hat_others']], dim=1)
    ##########################

    # Plot static map elements and actor tracks
    _plot_static_map_elements(data_batch['lane_positions'], data_batch['lane_padding_mask'])

    veh_mask = (data_batch['x_attr']==1)[...,0]
    cur_plot_bounds = _plot_actor_tracks(ax, data_batch['x'][veh_mask].cpu(), data_batch['x_positions'][veh_mask].cpu(), data_batch['x_angles'][veh_mask].cpu(), data_batch['y'][veh_mask].cpu(), data_batch['x_padding_mask'][veh_mask].cpu(), historical_steps)
    plot_bounds = cur_plot_bounds

    # prediction = y_hat[0,0][data_batch['x_padding_mask'][0,0,historical_steps:]].unsqueeze(0).detach().cpu()
    prediction = output['y_hat'][0][:,~data_batch['x_padding_mask'][0,0,historical_steps:]].detach().cpu().numpy()
    if prediction is not None:
        _scatter_polylines(
            prediction[:, :, :],
            ax,
            color="mediumblue",
            grad_color=False,
            alpha=0.8,
            scale=0.5,
            linewidth=2,
            zorder=1000,
            headwidth=1,
            headlength=1,
        )

    plt.axis("equal")
    plt.xlim(
        plot_bounds[0] - _PLOT_BOUNDS_BUFFER_W, plot_bounds[0] + _PLOT_BOUNDS_BUFFER_W
    )
    plt.ylim(
        plot_bounds[1] - _PLOT_BOUNDS_BUFFER_H, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_H
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)
    else:
        plt.show()

    plt.close('all')

def visualize_mae(
    # scenario: ArgoverseScenario,
    # scenario_static_map: ArgoverseStaticMap,
    # prediction: np.ndarray = None,
    # timestep: int = 50,
    data_batch: Dict,
    output: Dict,
    historical_steps,
    save_path: Path = None,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[0].set_facecolor('xkcd:grey')
    ax[1].set_facecolor('xkcd:grey')

    _plot_static_map_elements(data_batch['lane_positions'], data_batch['lane_padding_mask'], ax=ax[0], lane_pred_mask=output['lane_pred_mask'])
    _plot_static_map_elements(data_batch['lane_positions'], data_batch['lane_padding_mask'], ax=ax[1], lane_pred_mask=output['lane_pred_mask'], lane_mae_pred=output['lane_mae_pred']+data_batch['lane_centers'].unsqueeze(-2))

    veh_mask = (data_batch['x_attr']==1)[...,0]
    cur_plot_bounds = _plot_actor_tracks_mae(ax[0], 
                                             data_batch['x'][veh_mask].cpu(), 
                                             data_batch['x_positions'][veh_mask].cpu(), 
                                             data_batch['x_angles'][veh_mask].cpu(), 
                                             data_batch['y'][veh_mask].cpu(), 
                                             data_batch['x_padding_mask'][veh_mask].cpu(), 
                                             historical_steps,
                                             hist_pred_mask=output['hist_pred_mask'][veh_mask].cpu(),
                                             future_pred_mask=output['future_pred_mask'][veh_mask].cpu(),
                                             )
    x_mae_hat = output['x_mae_hat'][veh_mask]# + data_batch['x_centers'][veh_mask].unsqueeze(-2)
    y_mae_hat = output['y_mae_hat'][veh_mask]# + data_batch['x_centers'][veh_mask].unsqueeze(-2)
    cur_plot_bounds = _plot_actor_tracks_mae(ax[1], 
                                             data_batch['x'][veh_mask].cpu(), 
                                             data_batch['x_positions'][veh_mask].cpu(), 
                                             data_batch['x_angles'][veh_mask].cpu(), 
                                             data_batch['y'][veh_mask].cpu(), 
                                             data_batch['x_padding_mask'][veh_mask].cpu(), 
                                             historical_steps,
                                             hist_pred_mask=output['hist_pred_mask'][veh_mask].cpu(),
                                             future_pred_mask=output['future_pred_mask'][veh_mask].cpu(),
                                             x_mae_hat=x_mae_hat.detach().cpu(),
                                             y_mae_hat=y_mae_hat.detach().cpu(),
                                             )
    
    plot_bounds = cur_plot_bounds

    ax[0].set_aspect("equal")
    ax[0].set_xlim(
        plot_bounds[0] - _PLOT_BOUNDS_BUFFER_W, plot_bounds[0] + _PLOT_BOUNDS_BUFFER_W
    )
    ax[0].set_ylim(
        plot_bounds[1] - _PLOT_BOUNDS_BUFFER_H, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_H
    )
    
    ax[1].set_aspect("equal")
    ax[1].set_xlim(
        plot_bounds[0] - _PLOT_BOUNDS_BUFFER_W, plot_bounds[0] + _PLOT_BOUNDS_BUFFER_W
    )
    ax[1].set_ylim(
        plot_bounds[1] - _PLOT_BOUNDS_BUFFER_H, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_H
    )

    plt.tight_layout()
    plt.title(save_path.split('/')[-1].split('.')[0])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)
    else:
        print('no save path')
        # plt.show()

    plt.close('all')


def _plot_static_map_elements(
    lane_positions,
    lane_padding_mask,
    **kwargs,
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # # Plot drivable areas
    # for drivable_area in static_map.vector_drivable_areas.values():
    #     _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for li, lane_segment in enumerate(lane_positions[0]):
        lane_padding = lane_padding_mask[0,li]
        centerline = lane_segment[~lane_padding].cpu()
        
        # centerline = static_map.get_lane_segment_centerline(lane_segment.id)
        if 'ax' in kwargs.keys():

            if 'lane_pred_mask' in kwargs.keys() and 'lane_mae_pred' in kwargs.keys() and kwargs['lane_pred_mask'][0,li]:
                centerline_pred = kwargs['lane_mae_pred'][0,li][~lane_padding].detach().cpu()
                _plot_polylines(
                    [centerline_pred], line_width=3, color="white", endpoint=False, zorder=98, ax=kwargs['ax']
                )
            else:
                _plot_polylines(
                    [centerline], line_width=2.0, color="#000000", alpha=0.2, style="--", ax=kwargs['ax']
                )

            if 'lane_pred_mask' in kwargs.keys() and kwargs['lane_pred_mask'][0,li]:
                continue

            _plot_polylines(
                [centerline],
                line_width=3,
                color="white",
                endpoint=False,
                zorder=98, 
                ax=kwargs['ax']
            )



        else:
            _plot_polylines(
                [centerline], line_width=2.0, color="#000000", alpha=0.2, style="--"
            )
            _plot_polylines(
                [centerline],
                line_width=3,
                color="white",
                endpoint=False,
                zorder=98,
            )

    


def _plot_actor_tracks(
    # ax: plt.Axes,
    # scenario,#: ArgoverseScenario,
    # timestep: int,
    ax,
    x,
    x_positions,
    x_angles,
    y,
    padding_mask,
    historical_steps
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    # focal_id = scenario.focal_track_id

    for ai in range(x.size(0)):
        padding_mask_ai = padding_mask[ai]

        future_trajectory = np.array(y[ai][~padding_mask_ai[historical_steps:]] + x_positions[ai, historical_steps-1].unsqueeze(0))
        history_trajectory = np.array(x[ai][~padding_mask_ai[:historical_steps]] + x_positions[ai, historical_steps-1].unsqueeze(0))

        actor_headings = np.array(x_angles[ai,:historical_steps][~padding_mask_ai[:historical_steps]])

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        # if track.category == TrackCategory.FOCAL_TRACK:
        if ai == 0:
            _scatter_polylines(
                [future_trajectory],
                cmap="autumn",
                linewidth=6,
                reverse=True,
                arrow=True,
                scale=0.25,
                ax=ax,
            )
        # elif track.object_type in _STATIC_OBJECT_TYPES:
        #     continue

        track_color = _DEFAULT_ACTOR_COLOR
        _scatter_polylines([history_trajectory], cmap="Blues", linewidth=5, arrow=False, scale=0.25, ax=ax)
        if ai == 0:
            track_bounds = history_trajectory[-1]
            track_color = _FOCAL_AGENT_COLOR

        # Plot bounding boxes for all vehicles and cyclists
        # if track.object_type == ObjectType.VEHICLE:
        _plot_actor_bounding_box(
            ax,
            history_trajectory[-1],
            actor_headings[-1],
            track_color,
            (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
        )

    return track_bounds

def _plot_actor_tracks_all(
    # ax: plt.Axes,
    # scenario,#: ArgoverseScenario,
    # timestep: int,
    ax,
    x,
    x_positions,
    x_angles,
    y,
    padding_mask,
    historical_steps
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    # focal_id = scenario.focal_track_id
    Reds_r_tr = truncate_colormap(plt.get_cmap('Reds_r'), 0.2, 0.8)

    for ai in range(x.size(0)):
        padding_mask_ai = padding_mask[ai]

        future_trajectory = np.array(y[ai][~padding_mask_ai[historical_steps:]] + x_positions[ai, historical_steps-1].unsqueeze(0))
        history_trajectory = np.array(x[ai][~padding_mask_ai[:historical_steps]] + x_positions[ai, historical_steps-1].unsqueeze(0))

        actor_headings = np.array(x_angles[ai,:historical_steps][~padding_mask_ai[:historical_steps]])

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        # if track.category == TrackCategory.FOCAL_TRACK:
        if ai == 0:
            _scatter_polylines(
                [future_trajectory],
                cmap=Reds_r_tr,
                linewidth=6,
                reverse=True,
                arrow=True,
                scale=0.25,
                alpha=0.6,
                ax=ax,
            )
        # elif track.object_type in _STATIC_OBJECT_TYPES:
        #     continue

        track_color = _DEFAULT_ACTOR_COLOR
        _scatter_polylines([history_trajectory], cmap="Blues", linewidth=6, arrow=False, alpha=0.6, scale=0.25, ax=ax)
        if ai == 0:
            track_bounds = history_trajectory[-1]
            track_color = _FOCAL_AGENT_COLOR

        # Plot bounding boxes for all vehicles and cyclists
        # if track.object_type == ObjectType.VEHICLE:
        _plot_actor_bounding_box(
            ax,
            history_trajectory[-1],
            actor_headings[-1],
            track_color,
            (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
        )

    return track_bounds

def _plot_actor_tracks_mae(
    # ax: plt.Axes,
    # scenario,#: ArgoverseScenario,
    # timestep: int,
    ax,
    x,
    x_positions,
    x_angles,
    y,
    padding_mask,
    historical_steps,
    hist_pred_mask,
    future_pred_mask,
    x_mae_hat=None,
    y_mae_hat=None,
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    # focal_id = scenario.focal_track_id
    Reds_r_tr = truncate_colormap(plt.get_cmap('Reds_r'), 0.2, 0.5)
    Blues_tr = truncate_colormap(plt.get_cmap('Blues_r'), 0.2, 0.5)
    Grays_tr = truncate_colormap(plt.get_cmap('gray'), 0.2, 0.5)

    for ai in range(x.size(0)):
        padding_mask_ai = padding_mask[ai]

        future_trajectory = np.array(y[ai][~padding_mask_ai[historical_steps:]] + x_positions[ai, historical_steps-1].unsqueeze(0))
        history_trajectory = np.array(x_positions[ai, :historical_steps][~padding_mask_ai[:historical_steps]])

        actor_headings = np.array(x_angles[ai,:historical_steps][~padding_mask_ai[:historical_steps]])

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        # if track.category == TrackCategory.FOCAL_TRACK:
        # if ai == 0:
        
        if y_mae_hat is None:
            if future_pred_mask[ai]:
                fut_cmap = Grays_tr
            else:
                fut_cmap = Reds_r_tr
            
            _scatter_polylines(
                [future_trajectory],
                cmap=fut_cmap,
                linewidth=6,
                reverse=True,
                arrow=True,
                scale=0.25,
                alpha=0.9,
                ax=ax,
            )
        else:
            fut_cmap = Reds_r_tr
            if future_pred_mask[ai]:
                future_trajectory = np.array(y_mae_hat[ai][~padding_mask_ai[historical_steps:]] + x_positions[ai, historical_steps-1].unsqueeze(0))

            _scatter_polylines(
                [future_trajectory],
                cmap=fut_cmap,
                linewidth=6,
                reverse=True,
                arrow=True,
                scale=0.25,
                alpha=0.9,
                ax=ax,
            )

        # elif track.object_type in _STATIC_OBJECT_TYPES:
        #     continue

        track_color = _DEFAULT_ACTOR_COLOR

        if x_mae_hat is None:
            if hist_pred_mask[ai]:
                hist_cmap = Grays_tr
            else:
                hist_cmap = Blues_tr
            _scatter_polylines([history_trajectory], cmap=hist_cmap, linewidth=6, arrow=False, alpha=0.9, scale=0.25, ax=ax)
        else:
            hist_cmap = Blues_tr
            if hist_pred_mask[ai]:
                history_trajectory = np.array(x_mae_hat[ai][~padding_mask_ai[:historical_steps]] + x_positions[ai, historical_steps-1].unsqueeze(0))
            _scatter_polylines([history_trajectory], cmap=hist_cmap, linewidth=6, arrow=False, alpha=0.9, scale=0.25, ax=ax)
        
        if ai == 0:
            track_bounds = history_trajectory[-1]
            # track_color = _FOCAL_AGENT_COLOR

        # Plot bounding boxes for all vehicles and cyclists
        # if track.object_type == ObjectType.VEHICLE:
        # _plot_actor_bounding_box(
        #     ax,
        #     history_trajectory[-1],
        #     actor_headings[-1],
        #     track_color,
        #     (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
        # )

    return track_bounds


class HandlerColorLineCollection(HandlerLineCollection):
    def __init__(
        self,
        reverse: bool = False,
        marker_pad: float = ...,
        numpoints: None = ...,
        **kwargs,
    ) -> None:
        super().__init__(marker_pad, numpoints, **kwargs)
        self.reverse = reverse

    def create_artists(
        self, legend, artist, xdescent, ydescent, width, height, fontsize, trans
    ):
        x = np.linspace(0, width, self.get_numpoints(legend) + 1)
        y = np.zeros(self.get_numpoints(legend) + 1) + height / 2.0 - ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap, transform=trans)
        lc.set_array(x if not self.reverse else x[::-1])
        lc.set_linewidth(artist.get_linewidth())
        return [lc]


def _plot_polylines(
    polylines,#: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
    endpoint: bool = False,
    ax=None,
    **kwargs,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    if ax is not None:
        for polyline in polylines:
            ax.plot(
                polyline[:, 0],
                polyline[:, 1],
                style,
                linewidth=line_width,
                color=color,
                alpha=alpha,
                **kwargs,
            )
            if endpoint:
                ax.scatter(polyline[0, 0], polyline[0, 1], color=color, s=15, **kwargs)
    else:
        for polyline in polylines:
            plt.plot(
                polyline[:, 0],
                polyline[:, 1],
                style,
                linewidth=line_width,
                color=color,
                alpha=alpha,
                **kwargs,
            )
            if endpoint:
                plt.scatter(polyline[0, 0], polyline[0, 1], color=color, s=15, **kwargs)


def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_lane(xy: np.ndarray, arc_length: np.ndarray, steps: np.ndarray):
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def interpolate_centerline(xy: np.ndarray, n_points: int):
    arc_length = get_polyline_arc_length(xy)
    steps = np.linspace(0, arc_length[-1], n_points)
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def _scatter_polylines(
    polylines,#: Sequence[NDArrayFloat],
    cmap="spring",
    linewidth=3,
    arrow: bool = True,
    reverse: bool = False,
    alpha=0.5,
    zorder=100,
    scale=0.25,
    grad_color: bool = True,
    color=None,
    linestyle="-",
    headwidth=None,
    headlength=None,
    ax=None,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    if ax is None:
        ax = plt.gca()
    for polyline in polylines:
        inter_poly = interpolate_centerline(polyline, 50)

        if arrow:
            point = inter_poly[-1]
            diff = inter_poly[-1] - inter_poly[-2]
            diff = diff / np.linalg.norm(diff)
            if grad_color:
                c = plt.cm.get_cmap(cmap)(0)
            else:
                c = color
            if headwidth is None:
                arrow = ax.quiver(
                    point[0],
                    point[1],
                    diff[0],
                    diff[1],
                    alpha=alpha,
                    scale_units="xy",
                    scale=scale,
                    minlength=0.1,
                    minshaft=0.01,
                    zorder=zorder - 1,
                    color=c,
                )
            else:
                arrow = ax.quiver(
                    point[0],
                    point[1],
                    diff[0],
                    diff[1],
                    alpha=alpha,
                    scale_units="xy",
                    scale=scale,
                    minlength=0.1,
                    minshaft=0.01,
                    zorder=zorder - 1,
                    color=c,
                    headwidth=headwidth,
                    headlength=headlength,
                )

        if grad_color:
            arc = get_polyline_arc_length(inter_poly)
            polyline = inter_poly.reshape(-1, 1, 2)
            segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
            norm = plt.Normalize(arc.min(), arc.max())
            lc = LineCollection(
                segment, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha
            )
            lc.set_array(arc if not reverse else arc[::-1])
            lc.set_linewidth(linewidth)
            ax.add_collection(lc)
        else:
            ax.plot(
                inter_poly[:, 0],
                inter_poly[:, 1],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                linestyle=linestyle,
            )


def _plot_polygons(
    polygons,#: Sequence[NDArrayFloat], 
    *, 
    alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(
            polygon[:, 0],
            polygon[:, 1],
            fc=to_rgba(_DRIVABLE_AREA_COLOR, alpha),
            ec="black",
            linewidth=2,
            zorder=2,
        )


def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location,#: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        np.degrees(heading),
        zorder=_BOUNDING_BOX_ZORDER + 100,
        fc=color,
        ec="dimgrey",
        alpha=1.0,
    )
    ax.add_patch(vehicle_bounding_box)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
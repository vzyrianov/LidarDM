import cv2
import numpy as np

from waymax import visualization

from .waymax_utils import *

__all__ = ["get_lidardm_map"]


def plot_bounding(scenario, use_log_traj=True, lock_first_frame=False, m_coverage=96, width=640):
  traj = scenario.log_trajectory if use_log_traj else scenario.sim_trajectory
  time_idx = scenario.timestep
  traj_5dof = np.array(
      traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
  )  # Forces to np from jnp

  traj_5dof = traj_5dof[get_ego_mask(scenario) == 0]

  if lock_first_frame:
    origin_x, origin_y, origin_yaw = get_ego_position(scenario.replace(timestep=0), use_log_traj)
  else:
    origin_x, origin_y, origin_yaw = get_ego_position(scenario, use_log_traj)

  num_obj, num_steps, _ = traj_5dof.shape
  if time_idx is not None:
    if time_idx == -1:
      time_idx = num_steps - 1
    if time_idx >= num_steps:
      raise ValueError('time_idx is out of range.')

  time_indices = np.tile(
      np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1)
  )

  traj_5dof = traj_5dof[(time_indices == time_idx)]
  traj_5dof[:,0] -= origin_x
  traj_5dof[:,1] -= origin_y
  traj_5dof[:,:2] = rotate_nx2_array(traj_5dof[:,:2], origin_yaw)
  traj_5dof[:,-1] -= origin_yaw

  traj_5dof[:,:4] = (traj_5dof[:,:4] * (width / m_coverage))
  traj_5dof[:,:2] = (traj_5dof[:,:2]) + (width / 2)

  renders = [
    np.zeros((width, width), dtype=np.uint8), # render_unknown 
    np.zeros((width, width), dtype=np.uint8), # render_vehicle 
    np.zeros((width, width), dtype=np.uint8), # render_pedestrian 
    np.zeros((width, width), dtype=np.uint8), # render_cyclist 
  ]

  obj_type = scenario.object_metadata.object_types
  obj_type = obj_type.at[obj_type == 4].set(0)
  obj_type = obj_type[get_ego_mask(scenario) == 0]

  for obj_idx in range(num_obj):
    box = traj_5dof[obj_idx]
    box = get_np_bbox(box).round().astype(np.int32)
    cv2.fillPoly(renders[obj_type[obj_idx]], [box], 1)

  res = (np.stack(renders, -1) > 0.5).astype(np.uint8)
  x = res
    
  x = np.rot90(x, 2, (0, 1))
  x = np.flip(x, 1)

  x = np.concatenate([np.zeros_like(x), x], axis=2)
  return x

def plot_map_features(scenario, use_log_traj=True, lock_first_frame=False, m_coverage=96, width=640):
  traj = scenario.log_trajectory if use_log_traj else scenario.sim_trajectory
  current_xy = traj.xy[:, scenario.timestep, :]
  viz_config = visualization.utils.VizConfig()
  
  grouped_ids = [
    np.arange(0, 4),    #lanes, 
    np.arange(5, 14),   #road_lines, 
    np.arange(14, 17),  #road_edges, 
    # np.array([17]),     #stop_signs, 
    np.array([18]),     #crosswalk, 
    # np.array([19]),     #speed_bump
    np.array([20])
  ]

  if lock_first_frame:
    origin_x, origin_y, origin_yaw = get_ego_position(scenario.replace(timestep=0), use_log_traj)
  else:
    origin_x, origin_y, origin_yaw = get_ego_position(scenario, use_log_traj)

  renders = [np.zeros((width, width), dtype=np.uint8) for _ in range(len(grouped_ids))]

  rg_pts = scenario.roadgraph_points
  xy = rg_pts.xy[rg_pts.valid]
  rg_type = rg_pts.types[rg_pts.valid]

  # do rotate
  xy = xy.at[:,0].add(-origin_x)
  xy = xy.at[:,1].add(-origin_y)
  xy = rotate_nx2_array(xy, origin_yaw)

  for curr_type in np.unique(rg_type):
    map_id = None
    for color_id in range(len(grouped_ids)):
      if curr_type in grouped_ids[color_id]:
        map_id = color_id
        break

    if map_id is None: continue
    p1 = xy[rg_type == curr_type]

    p1 = (p1 * (width / m_coverage)) + (width / 2)
    p1 = np.pad(p1, ((0, 1), (0, 0)), constant_values=0.0)        # 3 n
    p1 = np.pad(p1, ((0, 1), (0, 0)), constant_values=1.0)        # 4 n
    p1 = p1[:,:2].round().astype(np.int32)

    for i in range(len(p1)):
      cv2.circle(renders[map_id], (p1[i,0], p1[i,1]), 1, 1, thickness=4)

  x = (np.stack(renders) > 0.5).astype(np.uint8)
  
  # print(x.shape)
  x = np.rot90(x, 2, (1, 2))
  x = np.flip(x, 2)
  x = np.transpose(x, (1, 2, 0))
  
  return x

def get_lidardm_map(scenario, use_log_traj=True, lock_first_frame=False):
  figure = plot_map_features(scenario, use_log_traj, lock_first_frame=lock_first_frame)
  objects = plot_bounding(scenario, use_log_traj, lock_first_frame=lock_first_frame)

  return np.concatenate([figure, objects], 2)
import numpy as np
from waymax import visualization

def get_ego_mask(scenario):
  mask = np.zeros(scenario.num_objects)

  viz_config = visualization.utils.VizConfig()

  if viz_config.center_agent_idx == -1:
    mask[scenario.object_metadata.is_sdc] = 1
  else:
    mask[viz_config.center_agent_idx] = 1
  
  return mask

def get_np_bbox(bboxes: np.ndarray):

  c = np.cos(bboxes[4])
  s = np.sin(bboxes[4])
  pt = np.array((bboxes[0], bboxes[1]))  # (2, N)
  length, width = bboxes[2], bboxes[3]
  u = np.array((c, s))
  ut = np.array((s, -c))

  # Compute box corner coordinates.
  tl = pt + length / 2 * u - width / 2 * ut
  tr = pt + length / 2 * u + width / 2 * ut
  br = pt - length / 2 * u + width / 2 * ut
  bl = pt - length / 2 * u - width / 2 * ut
  
  return np.vstack((tl, tr, br, bl))

def rotate_nx2_array(array, angle):
  rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],[np.sin(-angle), np.cos(-angle)]])
  array = (rotation_matrix @ array.T).T
  return array

def get_ego_position(scenario, use_log_traj=True, time_idx=0):
  traj = scenario.log_trajectory if use_log_traj else scenario.sim_trajectory
  current_xy = traj.xy[:, scenario.timestep, :]
  current_yaw = traj.yaw[:, scenario.timestep]
  viz_config = visualization.utils.VizConfig()

  if viz_config.center_agent_idx == -1:
    xy = current_xy[scenario.object_metadata.is_sdc]
    yaw = current_yaw[scenario.object_metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
    yaw = current_yaw[viz_config.center_agent_idx]
  x, y = xy[time_idx, :2]
  yaw = yaw[time_idx]
  return x, y, yaw

def get_masked_position(scenario, use_log_traj=True, idx = 0):
  traj = scenario.log_trajectory if use_log_traj else scenario.sim_trajectory
  current_xy = traj.xy[:, scenario.timestep, :]
  current_yaw = traj.yaw[:, scenario.timestep]

  xy = current_xy[idx]
  yaw = current_yaw[idx]
  x, y = xy
  yaw = yaw
  return x, y, yaw

def x_y_yaw_to_pose(x, y, yaw):
  return np.array([[np.cos(yaw), -np.sin(yaw), 0,    x],
                  [np.sin(yaw), np.cos(yaw) , 0,    y],
                  [0          , 0           , 1,    0],
                  [0          , 0           , 0,    1]])

def get_3d_bbox(box):
  box_5dof = box[:5]
  bbox_2d = get_np_bbox(box_5dof)
  bbox_3d = np.vstack((bbox_2d, bbox_2d))

  z, h = box[-2:]
  z_values = np.array([0,0,0,0,h,h,h,h])[:, np.newaxis]
  return np.hstack((bbox_3d, z_values))
import numpy as np
import open3d as o3d
from typing import Any, Dict, List, Tuple

from natsort import natsorted

__all__ = ['transform_points_to_pose', 
           'get_relative_pose', 
           'convert_poses_to_relative',
           'get_box_center',
           'get_transformation_from_box_to_box',
           'apply_poses_to_box',
           'get_first_item_in_dict',
           'get_last_item_in_dict',
           'get_direction_from_bbox']

def get_box_center(box: np.ndarray) -> np.ndarray:
  x_min, x_max = np.min(box[:,0]), np.max(box[:,0])
  y_min, y_max = np.min(box[:,1]), np.max(box[:,1])
  z_min, z_max = np.min(box[:,2]), np.max(box[:,2])

  return np.array([(x_max + x_min) / 2, 
                   (y_max + y_min) / 2, 
                   (z_max + z_min) / 2])

def get_direction_from_bbox(box):
  x_min, x_max = np.min(box[:,0]), np.max(box[:,0])
  y_min, y_max = np.min(box[:,1]), np.max(box[:,1])
  heading = np.array([x_max - x_min,y_max - y_min])
  return heading

def get_2D_rotation_matrix_aligned_vector(v1, v2):
  a = v1 / np.linalg.norm(v1)
  b = v2 / np.linalg.norm(v2)

  return np.array([[a[0]*b[0] + a[1]*b[1], b[0]*a[1] - a[0]*b[1], 0],
                   [a[0]*b[1] - b[0]*a[1], a[0]*b[0] + a[1]*b[1], 0],
                   [0                    , 0                    , 1]])

def transform_points_to_pose(points, pose):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  pcd.transform(pose)

  return np.asarray(pcd.points)

def apply_poses_to_box(box, start_id, n_frames, poses):
  new_bboxes = {}

  current_id = start_id
  for frame in poses:
    new_bboxes[current_id] = transform_points_to_pose(box, poses[frame])
    current_id += 1
    if current_id >= n_frames:
      break

  return new_bboxes
   

def get_transformation_from_box_to_box(box1, box2):
  center_1 = get_box_center(box1) * [1,1,0]
  center_2 = get_box_center(box2) * [1,1,0]

  t = center_2-center_1
  R = get_2D_rotation_matrix_aligned_vector(get_direction_from_bbox(box2),
                                            get_direction_from_bbox(box1))

  T = np.hstack((np.eye(3), t[:,np.newaxis]))
  T = np.vstack((T, np.array([0,0,0,1])))
  return T

def get_relative_pose(pose: np.ndarray,
                      center_pose: np.ndarray) -> np.ndarray:
  return np.linalg.inv(center_pose) @ pose

def get_center_pose_from_list(poses: List[np.ndarray]) -> np.ndarray:
  return poses[len(poses) // 2]

def convert_poses_to_relative(poses: List[np.ndarray], relative='middle') -> List[np.ndarray]:
  '''
  Functionalities:
  - Transform a list of poses to the coordinates frame of the "relative" pose

  Inputs:
  - poses: the list of poses
  - relative: transform the list to the coordinates of "first" or "middle" index of the poses

  Outputs:
  - list of transformed poses, where pose at index "relative" should be np.eye(4)
  '''
  
  if relative == 'middle':
    anchor_pose = get_center_pose_from_list(poses)
  elif relative == 'first':
    anchor_pose = poses[0]

  for i in range(len(poses)):
    poses[i] = get_relative_pose(poses[i], anchor_pose)
  return poses

def interpolate_between_poses(a):
  pass

def convert_agents_to_relative(poses: List[np.ndarray],
                               agents: Dict[str, Dict[str, Any]],
                               relative='middle') -> Dict[int, Any]:
  
  if relative == 'middle':
    anchor_pose = get_center_pose_from_list(poses)
  elif relative == 'first':
    anchor_pose = poses[0]

  for i, pose in enumerate(poses):
    if i in dict:
      relative_pose = get_relative_pose(pose, anchor_pose)
      dict[i] = transform_points_to_pose(dict[i], relative_pose)

  return dict

def get_first_item_in_dict(dict: Dict[Any, Any]) -> Tuple[Any, Any]:
  '''
  Functionalities:
  - Get the first natsorted element of a dictionary 

  Inputs:
  - dict: the dictionary 

  Outputs:
  - the key and value of the first element
  '''
  start_id = natsorted(dict.keys())[0]
  return start_id, dict[start_id]

def get_last_item_in_dict(dict: Dict[Any, Any]) -> Tuple[Any, Any]:
  '''
  Functionalities:
  - Get the last natsorted element of a dictionary 

  Inputs:
  - dict: the dictionary 

  Outputs:
  - the key and value of the last element
  '''
  end = natsorted(dict.keys())[-1]
  return end, dict[end]
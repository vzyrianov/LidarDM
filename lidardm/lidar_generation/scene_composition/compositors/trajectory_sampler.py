import os
import random
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np

from ..utils.utils import *
from .scene_compositor import AgentDict, AgentInfo

def create_bbox_corners(length: float, width: float, height: float) -> np.ndarray:
  '''
  Functionalities:
  - create an axis-aligned bounding box with specified dimension, centered at (0,0)

  Outputs:
  - 8x3 array for 8 corners of the bounding box
  '''
  xs = np.array([-length//2,length//2])
  ys = np.array([-width//2,width//2])
  zs = np.array([0,height])
  points = np.vstack(np.meshgrid(xs,ys,zs)).reshape(3,-1).T
  return points

def get_agents_in_frame(frame_id: int, 
                        bbox_coord_file: str, 
                        bbox_ids_file: str,
                        pose: np.ndarray = None) -> AgentDict:
  '''
  Functionality:
  - get all the bboxes of agents for a given frame of a given scenario

  Inputs:
  - frame_id: the frame to get the bboxes from 
  - bbox_coord_file: the metadata file to get the bbox corners locations
  - bbox_ids_file: the metadata file to get the bbox agent ids and semantics
  - pose: if not None, apply transformation of the bounding box to the pose, 
          since bbox are in current_frame coordinates, this will transform it
          to the world coordinates

  Outputs:
  - an AgentDict but every agent only has 1 bbox
  '''
  # if no bbox files found -> no dynamic object for that frame
  if not os.path.isfile(bbox_coord_file) or not os.path.isfile(bbox_ids_file):
    return AgentDict()
  
  # read bbox and ids
  box_coords = np.load(bbox_coord_file)
  box_ids = np.genfromtxt(bbox_ids_file, delimiter=" ", dtype='str')

  # populates bbox corners, ids, and semantic label 
  agents = AgentDict()
  for i in range(len(box_ids)):
    agent_id = box_ids[i,0] if box_ids.ndim > 1 else box_ids[0]
    current_box = box_coords[8*i:8*i+8, :]

    if pose is not None:
      current_box = transform_points_to_pose(current_box, pose)

    if len(current_box) != 8: continue

    current_semantic = box_ids[i,1] if box_ids.ndim > 1 else box_ids[1]
    current_yaw = float(box_ids[i,2] if box_ids.ndim > 1 else box_ids[2])

    agents.insert_agent(agent_id=agent_id,
                        semantic=current_semantic,
                        bboxes={frame_id: (current_box, current_yaw)})
  return agents

def get_agents_in_n_frames(poses: List[np.ndarray],
                           frame_ids: List[int], 
                           bbox_folder: str) -> AgentDict:
  '''
  Functionalities: 
  - get all agents and their bboxes for a scenario over n frames

  Inputs:
  - poses: list of ego-vehicle poses
  - frame_ids: list of frame ids, specify which folder of bbox_folder to get data from
  - bbox_folder: the tfrecord folder of the desired scenario

  Outputs:
  - an AgentDict
  '''

  agents_dict = AgentDict()

  for i in range(len(frame_ids)):
    current_frame = frame_ids[i]

    # obtain the dictionary of all agents of the current frame
    bbox_coord_file = os.path.join(bbox_folder, 'coords', f'{current_frame}.npy')
    bbox_ids_file = os.path.join(bbox_folder, 'ids_heading', f'{current_frame}.txt')


    current_agents = get_agents_in_frame(i, bbox_coord_file, bbox_ids_file, poses[i])
        
    # if agents dict is empty, we assign it to the agents in first frame
    # for subssequent frames, if a dynamic object disappears, we simply remove it
    # Essentially, we only keep the agent that is visible for all frames  
    agents_dict.merge_agent_dict(i, current_agents.agents)

  return agents_dict

class WaymoTrajectoriesSampler():

  def __init__(self, root: str, relative: str = 'first') -> None:
    '''
    Functionalities:
    - sample trajectories from the prerprocessed Waymo Open Dataset 
    
    Inputs:
    - root: path to Waymo dataset
    - relative: if want the sampled trajectories to be centered at first or middle frame
    '''
    self.root = root
    self.relative = relative
    self.ego_traj = sorted(glob(os.path.join(self.root, "*", "*", "poses.txt")))
    self.agents_bbox = sorted(glob(os.path.join(self.root, "*", "*", "dynamic_bbox")))
    assert len(self.ego_traj) == len(self.agents_bbox)

  def sample_ego_poses(self, n_frames: int = 10) -> List[np.ndarray]:
    '''
    Functionalities:
    - sample ego-vehicle poses randomly for the given length

    Outputs:
    - list of ego poses, centered at self.relative
    '''
    from datetime import datetime
    random.seed(datetime.now().timestamp())

    # get a random frame start
    random_idx = random.choice(range(len(self.ego_traj)))
    poses = np.loadtxt(self.ego_traj[random_idx], delimiter=' ', dtype=float)
    potential_start_frame = random.choice(range(0, poses.shape[0] - n_frames))

    # get the n_frames poses following that start frame
    ego_poses = []
    for i in range(n_frames):
      current_frame = potential_start_frame + i
      ego_poses.append(poses[current_frame,:][1:].reshape(4,4))

    return convert_poses_to_relative(ego_poses, relative='first' if self.relative else 'middle')

  def sample_dynamic_objects(self, n_frames: int = 10) -> AgentDict:
    '''
    Functionalities:
    - sample for all dynamic objects of a random scenario, for n_frames length

    Outputs:
    - an AgentDict of all agents of a random scenario
    '''

    # also get a random start frame 
    random_idx = random.choice(range(len(self.ego_traj)))
    poses = np.loadtxt(self.ego_traj[random_idx], delimiter=' ', dtype=float)
    potential_start_frame = random.choice(range(0, poses.shape[0] - n_frames))
    
    frames_ids = []
    box_folder = self.agents_bbox[random_idx]
    sampled_agent_poses = []
    for i in range(n_frames):
      current_frame = potential_start_frame + i
      frames_ids.append(current_frame)
      sampled_agent_poses.append(poses[current_frame,:][1:].reshape(4,4))

    sampled_agent_poses = convert_poses_to_relative(sampled_agent_poses, relative='first' if self.relative else 'middle')

    return get_agents_in_n_frames(sampled_agent_poses, frames_ids, box_folder)
  
  # def sample_dynamics_objects_in_relative_bbox(self, n_frames: int) -> AgentDict:
  #   '''
  #   Functionality:
  #   - [experimental] sample pose sequence that is relative to the first 

  #   Outputs:
  #   - an AgentDict
  #   '''
  #   agents = self.sample_dynamic_objects(n_frames)
  #   acceptable_agents = {}

  #   for agent_id in agents:
  #     _, first_box = get_first_item_in_dict(agents[agent_id]['bbox'])

  #     temp_agent = agents[agent_id]
  #     temp_agent['trans_from_anchor'] = {}
  #     for frame in temp_agent['bbox']:
  #       current_box = temp_agent['bbox'][frame]
  #       if len(current_box) == 0:
  #         continue
  #       transformation = get_transformation_from_box_to_box(first_box, current_box)
  #       temp_agent['trans_from_anchor'][frame] = transformation
      
  #     acceptable_agents[agent_id] = temp_agent

  #   return acceptable_agents 
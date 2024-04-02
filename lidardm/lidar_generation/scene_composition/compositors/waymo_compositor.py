import os
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np

from .generative_compositor import GenerativeCompositor

from lidardm.lidar_generation.scene_composition.utils.raycast import *
from .trajectory_sampler import *
from .scene_compositor import SceneCompositor

class WaymoCompositor(SceneCompositor):
  def __init__(self, 
               mesh: o3d.geometry.TriangleMesh,
               sequence_id: str,        # metadata, name of the tfrecord folder
               anchor_frame_idx: int,   # metadata, index of the pose
               is_start_frame: bool = False,    # true: frame_idx =-> first frame, false: frame_idx -> middle frame
               n_frames: int = 15,
               waymo_dataset_root: str = '',
               raycaster_config: str ='waymo',
               is_sampling: bool = False) -> None:
    '''
    Functionalities:
    - Wrapper of SceneCompositor to generate LiDAR for a given Waymo sequence

    Inputs:
    - sequence_id: metadata, the name of the tfrecord folder (tfrecord-segment-....)
    - anchor_frame_idx: metadata, the index of the "anchored" pose 
    - is_start_frame: True if want anchor_frame_index to be the first frame of the generated sequence
                      False if want to be the middle frame, refers to implementation below
    - is_sampling: [experimental] sample trajectories instead of playback
    '''

    # read metadata 
    tfrecord_folder = glob(os.path.join(waymo_dataset_root, "*", sequence_id))[0]
    bbox_folder = os.path.join(tfrecord_folder, "dynamic_bbox")
    all_poses = np.loadtxt(os.path.join(tfrecord_folder, "poses.txt"), delimiter=' ', dtype=float)

    # instantiate scene compositor 
    super().__init__(mesh=mesh, 
                     raycaster_config=raycaster_config)

    # create list of frame indexes
    start_idx = max(0, anchor_frame_idx - n_frames // 2) if not is_start_frame else anchor_frame_idx
    end_idx = min(len(all_poses) - 1, anchor_frame_idx + n_frames // 2) if not is_start_frame else (min(len(all_poses) - 1, anchor_frame_idx + n_frames))
    frames_ids = range(start_idx, end_idx+1)
    
    # obtain the desired ego poses
    ego_poses = []
    for i in range(len(frames_ids)):
      current_frame = frames_ids[i]
      ego_poses.append(all_poses[current_frame,:][1:].reshape(4,4))
    ego_poses = convert_poses_to_relative(ego_poses, relative='first' if is_start_frame else 'middle')
    
    # get all agents
    agents = get_agents_in_n_frames(ego_poses, frames_ids, bbox_folder)
    
    # experimental: sample trajectories from waymo downloaded bank
    # if is_sampling:
    #   generator = GenerativeCompositor(waymo_dataset_root='',
    #                                    mesh=mesh,
    #                                    n_frames=n_frames,
    #                                    sample_only=True,
    #                                    raycaster_config='waymo',
    #                                    is_start_frame=is_start_frame)
    #   anchor_frame_id = anchor_frame_idx - frames_ids[0]
    #   generator.update_scenarios(ego_poses, {})
    #   agents = generator.sample_agents_from_known_poses(anchor_frame_id, 
    #                                                     agents,
    #                                                     ego_poses)

    super().update_scenarios(ego_poses, agents)

  def __len__(self) -> int:
    return len(self.poses)
  
  def __getitem__(self, idx) -> Dict[str, Any]:	
    return super().__getitem__(frame=idx)
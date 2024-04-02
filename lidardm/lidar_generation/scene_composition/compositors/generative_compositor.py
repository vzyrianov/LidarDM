import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from .scene_compositor import SceneCompositor
from .trajectory_sampler import *

from ..utils.utils import *
from lidardm import PROJECT_DIR
from lidardm.lidar_generation.scene_composition.utils.raycast import *

MOVEMENT_THRESHOLD = 5

class GenerativeCompositor(SceneCompositor):
  def __init__(self, 
               mesh: o3d.geometry.TriangleMesh, 
               n_objects: int = 5,
               n_frames: int = 10,
               sampling_threshold: int = 1000,
               raycaster_config: str = 'kitti360',
               is_start_frame: bool = False,
               waymo_dataset_root: str = os.path.join(PROJECT_DIR, '_datasets', 'waymo_preprocessed'),
               sample_only: bool = False) -> None:
    '''
    Functionalities:
    - sample from Waymo trajectories bank to compose the 4D world 
    - [experimental] sample trajectories for conditional waymo dataset as well

    Inputs:
    - sample_only: True if only want to use the trajectory sampling part, 
                   not the 4D world composition
    '''
  
    self.sampling_threshold = sampling_threshold 
    self.use_sampling_dataset = os.path.isdir(waymo_dataset_root)
    if self.use_sampling_dataset:
      self.scenario_sampler = WaymoTrajectoriesSampler(waymo_dataset_root)

    self.ego_bbox_corners = create_bbox_corners(length=8, width=6, height=2.5)
    self.n_frames = n_frames
    self.is_start_frame = is_start_frame
    
    super().__init__(mesh=mesh, raycaster_config=raycaster_config)
    if not sample_only:
      self._sample_scenario(n_objects, n_frames)
      super().update_scenarios(self.sampled_ego_poses, self.sampled_agents)

  def __len__(self) -> int:
    return len(self.poses)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    return super().__getitem__(frame=idx)

  def _sample_scenario(self,
                       n_objects: int = 5,
                       n_frames: int = 10) -> None:
    '''
    Functionalities:
    - generate the scenario for the generative compositor, i.e.
      sample ego poses as well as agent poses

    Inputs:
    - n_objects: the number of agents to sample for
    - n_frames: the number of frames to sample for
    '''

    if not self.use_sampling_dataset: 
      print("Waymo Dataset isn't found, use straight ego-poses")
      xs = np.arange(0, 0.5 * self.n_frames, 0.5)
      sampled_ego_poses = [ np.array([[1,0,0,x],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for x in xs]
      sampled_agents = AgentDict()

    else:
      sampled_ego_poses = self.scenario_sampler.sample_ego_poses(n_frames)
      
      ego_sampling_count = 0
      while not self._evaluated_ego_poses(sampled_ego_poses):
        sampled_ego_poses = self.scenario_sampler.sample_ego_poses(n_frames)
        ego_sampling_count += 1

        if ego_sampling_count == self.sampling_threshold:
          raise KeyError(f'Couldnt sample ego poses within {self.sampling_threshold} tries')
      
      print("Got ego poses")

      sampled_agents = AgentDict()
      agent_sampling_count = 0
      while len(sampled_agents) < n_objects:
        current_sampled_agents = self.scenario_sampler.sample_dynamic_objects(n_frames)
        
        for sampled_agents_id in current_sampled_agents.agents:
          current_sampled_agent_info = current_sampled_agents.get_agent(sampled_agents_id)
          if self._evaluated_dynamic_object(agent_info=current_sampled_agent_info, 
                                            agent_dicts=sampled_agents, 
                                            ego_poses=sampled_ego_poses):
            sampled_agents.insert_agent_info(agent_id=sampled_agents_id, 
                                             agent_info=current_sampled_agent_info)
            if len(sampled_agents.get_agent_ids()) == n_objects:
              break

        agent_sampling_count += 1

        if agent_sampling_count == self.sampling_threshold:
          raise KeyError(f'Couldnt sample {n_objects} objects within {self.sampling_threshold} tries')
        
      print(f'Got {len(sampled_agents)} agents!')
    self.sampled_ego_poses = sampled_ego_poses
    self.sampled_agents = sampled_agents

  def _evaluated_ego_poses(self, ego_poses: List[np.ndarray]) -> bool:
    '''
    Functionalities:
    - heuristics to determine if the sampled ego poses should be accepted
    '''
    # reject if lower than certain threshold distance
    dist = np.linalg.norm(ego_poses[-1][:2, 3] - ego_poses[0][:2, 3])
    if dist < MOVEMENT_THRESHOLD:
      return False

    # reject if collision
    for ego_pose in ego_poses:
      current_bbox_corners = copy.deepcopy(self.ego_bbox_corners)
      bbox_corners = transform_points_to_pose(current_bbox_corners, ego_pose)
      status, _ = super().evaluate_bounding_boxes(bbox_corners=bbox_corners)
      if not status: 
        return False

    return True
  
  def _evaluated_dynamic_object(self,
                                agent_info: AgentInfo, 
                                ego_poses: List[np.ndarray],
                                agent_dicts: AgentDict) -> bool:
    '''
    Functionalities:
    - heuristics to determine if the sampled agent should be accepted
    '''
    
    bboxes = agent_info.bboxes
    for frame in bboxes:
      if len(bboxes) != len(ego_poses):
        return False
      
      current_ego_pose = ego_poses[frame]
      current_bbox, _ = agent_info.get_bbox_at_frame(frame)
      if current_bbox is None: 
        return False
      
      # if collide with static world -> false
      status, _ = super().evaluate_bounding_boxes(current_bbox)
      if not status: 
        return False
      
      # if colliding with current ego car -> false
      ego_bbox_corners = copy.deepcopy(self.ego_bbox_corners)
      ego_bbox_corners = transform_points_to_pose(ego_bbox_corners, current_ego_pose)
      if self._check_bbox_collide_topdown(bbox1=ego_bbox_corners, bbox2=current_bbox):
        return False
      
      # if colliding with any of the accepted agents -> false
      for accepted_agent_id, accepted_agent_info in agent_dicts.get_agent_iter():
        accepted_bbox, _ = accepted_agent_info.get_bbox_at_frame(frame)
        if accepted_bbox is None: continue
        if self._check_bbox_collide_topdown(bbox1=accepted_bbox, bbox2=current_bbox):
          return False
        
    return True
  
  def _check_bbox_collide_topdown(self, bbox1: np.ndarray, bbox2: np.ndarray) -> bool:    
    '''
    Functionalities:
    - heuristics to check if the two bbox collide or not

    Inputs:
    - bbox1 and bbox2: 8x3 ndarray matrix for 8 corners 
    '''
    bbox1_x_bound = np.min(bbox1[:,0]), np.max(bbox1[:,0])
    bbox1_y_bound = np.min(bbox1[:,1]), np.max(bbox1[:,1])
    bbox2_x_bound = np.min(bbox2[:,0]), np.max(bbox2[:,0])
    bbox2_y_bound = np.min(bbox2[:,1]), np.max(bbox2[:,1])

    def check_corner_inside_bbox(bbox1_x_bound, bbox1_y_bound, corner):
      if corner[0] <= bbox1_x_bound[1] and corner[0] >= bbox1_x_bound[0] and \
        corner[1] <= bbox1_y_bound[1] and corner[1] >= bbox1_y_bound[0]:
        return True
      return False
    
    corner_1 = [bbox2_x_bound[0], bbox2_y_bound[0]]
    corner_2 = [bbox2_x_bound[0], bbox2_y_bound[1]]
    corner_3 = [bbox2_x_bound[1], bbox2_y_bound[0]]
    corner_4 = [bbox2_x_bound[1], bbox2_y_bound[1]]

    return check_corner_inside_bbox(bbox1_x_bound, bbox1_y_bound, corner_1) or \
        check_corner_inside_bbox(bbox1_x_bound, bbox1_y_bound, corner_2) or \
        check_corner_inside_bbox(bbox1_x_bound, bbox1_y_bound, corner_3) or \
        check_corner_inside_bbox(bbox1_x_bound, bbox1_y_bound, corner_4) 
  
  # def sample_agents_from_known_poses(self, 
  #                                    agents: AgentDict, 
  #                                    ego_poses: List[np.ndarray]) -> AgentDict:
    
  #   agents_copy = copy.deepcopy(agents)
  #   new_agents = {}

  #   count = 0
  #   while len(agents_copy) > 0:
      
  #     # sampled for a trajectory
  #     sampled_agents = self.scenario_sampler.sample_dynamics_objects_in_relative_bbox(self.n_frames)
      
  #     # check if the semantic of the sampled agent match currenta agents
  #     for sampled_agent_id in sampled_agents:
  #       for agent_id in agents_copy:
  #         if sampled_agents[sampled_agent_id]['type'] == agents_copy[agent_id]['type']:
            
  #           # check if the first frame (frame that matches layout) satisfy
  #           start_id, anchor_box = get_first_item_in_dict(agents_copy[agent_id]['bbox'])
  #           status, _ = super().evaluate_bounding_boxes(anchor_box)
  #           if not status: 

  #             # not satisfy -> remove from desired agent
  #             del agents_copy[agent_id]
  #             print(f'Got {len(new_agents)} vs {len(agents_copy)} agents!')
  #             break

  #           # apply the trajectory to that first frame pose
  #           new_bbox = apply_poses_to_box(anchor_box, 
  #                                         start_id,
  #                                         self.n_frames,
  #                                         sampled_agents[sampled_agent_id]['trans_from_anchor'])

  #           # evaluate the trajectory 
  #           if self._evaluated_dynamic_object(bboxes=new_bbox, 
  #                                             agent_dicts=new_agents, 
  #                                             ego_poses=ego_poses):
            
  #             new_agents[agent_id] = agents_copy[agent_id]
  #             new_agents[agent_id]['bbox'] = new_bbox
  #             del agents_copy[agent_id]
  #             print(f'Got {len(new_agents)} vs {len(agents_copy)} agents!')
  #             break

  #     # break if too much iteration has passed thru
  #     count += 1
  #     if count == 25:
  #       break
      
  #   for agent_id in agents.get_agent_ids():
  #     start_id, (first_box, ) = get_first_item_in_dict(agents.get_agent(agent_id))
      
  #     status, _ = super().evaluate_bounding_boxes(anchor_box)
  #     if not status: 

  #       # not satisfy -> remove from desired agent
  #       del agents_copy[agent_id]
  #       print(f'Got {len(new_agents)} vs {len(agents_copy)} agents!')
  #       break

  #     sampled_poses = self.scenario_sampler.sample_ego_poses(self.n_frames)
  #     while 
      

  #   return new_agents
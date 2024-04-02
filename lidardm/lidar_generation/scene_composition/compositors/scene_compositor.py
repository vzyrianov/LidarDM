import open3d as o3d
import numpy as np

from typing import Any, Dict, List, Tuple

from ..utils.raycast import Raycaster, KITTI_RAYCAST_LARGE_CONFIG, WAYMO_RAYCAST_CONFIG
from ..utils.utils import *
from ..agents.agent import Vehicle, Pedestrian

__all__ = ["SceneCompositor", "AgentInfo", "AgentDict"]

class AgentInfo():
  '''
  Class that contains agent information for compositors
  '''
  def __init__(self, 
               semantic: str,
               bboxes: Dict[int, Tuple[np.ndarray, float]] = {}) -> None:
    self.semantic = semantic
    self.bboxes = bboxes

    if self.semantic == 'Pedestrian': 
      self.mesher = Pedestrian()
      self.color = [0,0,1] # paints pedestrians blue
    elif self.semantic == 'Vehicle':
      self.mesher = Vehicle()
      self.color = [1,0,0] # paints vehicle red
    else:
      raise KeyError(f'{self,semantic} not yet supported') 
    
  def update_bbox(self, frame: int, bbox: Tuple[np.ndarray, float]) -> None:
    if not isinstance(bbox, tuple): raise ValueError("bbox needs to be a tuple of [8x3 box corners, yaw]")
    self.bboxes[frame] = bbox

  def merge_info(self, bboxes: Dict[int, np.ndarray]):
    self.bboxes = self.bboxes | bboxes

  def get_bbox_at_frame(self, frame: int) -> Tuple[np.ndarray, float]:
    if frame not in self.bboxes:
      return None, None
    return self.bboxes[frame]

class AgentDict():
  '''
  Class that holds all agent information for a given scene, 
  essentially a dictionary of AgentInfo
  '''
  def __init__(self) -> None:
    self.agents = {}

  def insert_agent_info(self, agent_id: str, agent_info: AgentInfo) -> None:
    if agent_id in self.agents:
      self.agents[agent_id].merge_info(agent_info.bboxes)
    self.agents[agent_id] = agent_info

  def insert_agent(self, agent_id: str, semantic: str, bboxes: Dict[int, Tuple[np.ndarray, float]]) -> None:
    new_agent_info = AgentInfo(semantic, bboxes)
    self.insert_agent_info(agent_id, new_agent_info)

  def get_agent_ids(self) -> List[str]:
    return list(self.agents.keys())

  def get_agent_iter(self) -> Tuple[Any, Any]:
    return self.agents.items()
  
  def get_agent(self, agent_id: str) -> AgentInfo:
    return self.agents[agent_id]

  def merge_agent_dict(self, frame: int, other_agent_dict: Dict[str, AgentInfo]) -> None:
    for agent_id in self.get_agent_ids():
      if agent_id in other_agent_dict:
        self.agents[agent_id].merge_info(other_agent_dict[agent_id].bboxes)
        del other_agent_dict[agent_id]
      else:
        _, latest_box = get_last_item_in_dict(self.agents[agent_id].bboxes)
        self.agents[agent_id].update_bbox(frame, latest_box) 
    
    if len(other_agent_dict) > 0:
        self.agents = self.agents | other_agent_dict
  
  def __len__(self):
    return self.agents.__len__()
    

class SceneCompositor():
  '''
  A base Scene Compositor class
  ''' 
  def __init__(self, 
               raycaster_config: str, 
               mesh: o3d.geometry.TriangleMesh) -> None:

    # raycaster
    if raycaster_config == 'kitti360':
      self.raycaster = Raycaster(**KITTI_RAYCAST_LARGE_CONFIG)
    elif raycaster_config == 'waymo':
      self.raycaster = Raycaster(**WAYMO_RAYCAST_CONFIG)
    else:
      raise KeyError(f'{raycaster_config} not supported')
    self.extrinsic = self.raycaster.extrinsic

    # mesh
    if(type(mesh) is not o3d.t.geometry.TriangleMesh):
      self.world = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    else:
      self.world = mesh 
    self.height_map = self._compute_height_map(0.1)

  def update_scenarios(self, 
                       poses: List[np.ndarray], 
                       agents: AgentDict) -> None:
    '''
    Functionalities:
    - assigning class variables, written as a function for readability

    Inputs:
    - poses: List of SE3 4x4 ndarray matrix, the length of this list 
             determines the length of the frames
    - agents: an AgentDict where each agent's bboxes has the same frame index
              as the index of poses
    '''
    self.poses = poses
    self.agents = agents
    
  def __len__(self) -> int:
    return len(self.poses)

  def __getitem__(self, frame: int) -> Dict[str, Any]:
    '''
    Functionality: 
    - Compose the 3D world at given frame

    Inputs:
    - frame: frame index, needs to be within range(len(poses))

    Outputs:
    - Dict that contains LiDAR points, current_pose, agent mesh, etc...
    '''
    
    # create a raycasting scene
    self.scene = o3d.t.geometry.RaycastingScene()
    self.scene.add_triangles(self.world)
    agents = self._generate_agents(frame)

    # cast rays + decode hit-points
    pose = self.poses[frame] @ self.extrinsic
    pcd = self._generate_raycast(pose)

    return {
      "scan": pcd,
      "agents": agents,
      "pose": self.poses[frame],
    }
  
  def evaluate_bounding_boxes(self, bbox_corners: np.ndarray) -> Tuple[bool, o3d.geometry.OrientedBoundingBox]:
    '''
    Functionalities: 
    - determine if bounding box is on a planar surface
    - adjust the bounding box to be correctly placed on the ground

    Inputs:
    - bbox_corners: 8 corners of the bounding box, in shape of 8x3

    Outputs:
    - bool to determine if bounding box is valid or not
    - if valid, return the corrected bounding box as a 
    '''
    
    if len(bbox_corners) == 0:
      return False, None
    
    # get bounds of the bounding box to 
    x_min, x_max = np.min(bbox_corners[:,0]), np.max(bbox_corners[:,0])
    y_min, y_max = np.min(bbox_corners[:,1]), np.max(bbox_corners[:,1])
    z_min, z_max = np.min(bbox_corners[:,2]), np.max(bbox_corners[:,2])

    # obtain the height map within the bounding box
    points_in_hmap_mask = np.where(np.logical_and(self.height_map[:,0] > x_min, self.height_map[:,0] < x_max), True, False)
    points_in_hmap_mask &= np.where(np.logical_and(self.height_map[:,1] > y_min, self.height_map[:,1] < y_max), True, False)
    hmap_points_in_bbox = self.height_map[points_in_hmap_mask]

    # if no hmap points in the box, the object is off the mesh 
    if len(hmap_points_in_bbox) < 10: 
      return False, None

    # obtain ground level + reject mesh with high ground level compared to bboox
    ground_height, max_height = np.min(hmap_points_in_bbox[:,2]), np.max(hmap_points_in_bbox[:,2])
    if abs(ground_height - max_height) >= 1.5 or ground_height > 3:
      return False, None

    # cut bounding box at ground level
    bbox_corners[:,2] += ground_height - z_min
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_corners))

    return True, bbox

  def _generate_raycast(self, pose: np.ndarray) -> np.ndarray:
    '''
    Functionalities:
    - raycast from the specified pose 
    '''
    rays = self.raycaster.generate_rays(pose)
    ans = self.scene.cast_rays(rays)
      
    return self.raycaster.decode_hitpoints(ans['t_hit'].numpy()) 

  def _generate_agents(self, frame: int):
    '''
    Functionalities:
    - Obtain all the agent meshes at the given frames
    - Add the meshes to raycasting scene 

    Outputs:
    - Dictionary that maps agent_id to current mesh and current bbox
    '''
    agents = {}
    for agent_id, current_agent in self.agents.get_agent_iter():
      current_bbox, current_yaw = current_agent.get_bbox_at_frame(frame)
      if current_bbox is None: continue

      # check if region of bbox is valid
      status, bbox = self.evaluate_bounding_boxes(bbox_corners=current_bbox)
      if not status: continue

      # yaw = None
      # if 'yaw' in self.agents[agent_id]:
      #   yaw = self.agents[agent_id]['yaw'][frame]

      # accept + create mesh
      agent_mesh = current_agent.mesher.get_mesh_at_bbox(frame, bbox, current_yaw)
      agent_mesh.paint_uniform_color(current_agent.color)
      self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(agent_mesh))

      agents[agent_id] = {}
      agents[agent_id]['mesh'] = agent_mesh
      agents[agent_id]['bbox'] = bbox

    return agents

  def _compute_height_map(self, resolution: float) -> np.ndarray:
    '''
    Functionalities:
    - height_map is used to determine planar surface of the given world. It is being computed 
      by raycasting from infinitely far away and all the hitpoints are saved as hmap points
    
    Inputs:
    - resolution: the size of the grid of hmap points

    Outputs:
    - np.ndarray that holds all the hmap points of the given world
    '''

    # raycast scene
    heightmap_scene = o3d.t.geometry.RaycastingScene()
    heightmap_scene.add_triangles(self.world)

    # create xy unifrom grid
    x_max, y_max = self.world.get_max_bound().numpy()[:2] 
    x_min, y_min = self.world.get_min_bound().numpy()[:2]
    xs = np.arange(x_min, x_max, resolution)
    ys = np.arange(y_min, y_max, resolution)
    xy = np.vstack(np.meshgrid(xs,ys)).reshape(2, -1).T

    # create origin + direction -> ray
    origins = np.hstack((xy, np.repeat(100, xy.shape[0])[:,np.newaxis]))
    dirs = np.tile(np.array([0,0,-1]), (len(origins), 1))
    rays = np.hstack((origins, dirs))

    # cast rays + mask invalid rays
    o3d_rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = heightmap_scene.cast_rays(o3d_rays)['t_hit'].numpy()
    mask = np.where(np.logical_and(hits != np.inf, hits < 200), 1, 0)
    rays, hits = rays[mask == 1], hits[mask == 1]
    
    height_map = rays[:,:3] + rays[:,-3:] * hits[:, np.newaxis]

    return height_map
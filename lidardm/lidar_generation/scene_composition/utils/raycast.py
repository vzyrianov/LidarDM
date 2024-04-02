
import numpy as np
import open3d as o3d

from ..utils.utils import transform_points_to_pose
from .raycast_configs import *

from lidardm.visualization.range_consistency import *
from lidardm.lidar_generation.raydropping.utils.infer import RayDropInferer

class Raycaster():
  def __init__(self, azimuth_range: np.ndarray,   
               azimuth_res: float, 
               elevation_range: np.ndarray,
               elevation_beams: int,
               max_range: float,
               dataset: str, 
               extrinsic: np.ndarray):
    
    '''
    Inputs: azimuth_range: [min, max] [degree] 
        azimuth_res: [degree]
        elevation_range: [min, max] [degree]
        elevation_beams: [number of beams]
        range: [m]
    '''
    self.azimuth_range = azimuth_range * np.pi / 180.0 # radian
    self.azimuth_res = azimuth_res * np.pi / 180.0 # radian
    self.elevation_range = elevation_range * np.pi / 180.0 # radian
    self.elevation_res = (self.elevation_range[1] - self.elevation_range[0]) / elevation_beams # radian

    self.elevation_beams = elevation_beams
    self.max_range = max_range
    
    self.dataset = dataset
    self.directions = self._generate_directions()
    self.raydropper = RayDropInferer(dataset, threshold=0.2)

    self.extrinsic = extrinsic 

  def _generate_directions(self):
    azi = np.arange(self.azimuth_range[0], self.azimuth_range[1], self.azimuth_res)
    
    # waymo has weird inclination angle setup
    if self.dataset == 'waymo':
      ele = WAYMO_INCLINATIONS
    else: 
      ele = np.arange(self.elevation_range[0], self.elevation_range[1], self.elevation_res)

    assert len(ele) == self.elevation_beams, f'number of beams is not {self.elevation_beams}'
    
    # create meshgrid of all possible combination of azi and ele
    ae = np.vstack(np.meshgrid(azi,ele)).reshape(2,-1)

    directions = np.vstack((np.cos(ae[1,:]) * np.cos(ae[0,:]), 
                np.cos(ae[1,:]) * np.sin(ae[0,:]),
                np.sin(ae[1,:]))).T
    directions = directions / np.linalg.norm(directions, axis=1)[:,np.newaxis]
    return directions

  def generate_rays(self, pose: np.ndarray):
    '''
    xyz = location of the lidar
    '''
    self.pose = pose
    xyz = pose[:3,3]
    origins = np.tile(xyz, (len(self.directions), 1))
    self.rays = np.hstack((origins, self.directions))
    return o3d.core.Tensor(self.rays, dtype=o3d.core.Dtype.Float32)
  
  def _raydrop(self, pcd):
    
    # point cloud -> range iamge
    if self.dataset == 'waymo':
      raycast = transform_points_to_pose(pcd, np.linalg.inv(self.pose))
      raycast_im = project_range(raycast, **WAYMO_RANGE_CONFIG)
    elif self.dataset == 'kitti360':
      pcd = transform_points_to_pose(pcd, np.linalg.inv(self.pose))
      raycast_im = project_range(pcd, **KITTI_RANGE_CONFIG)

    # infer from model + apply mask
    gumbel_mask = self.raydropper.infer(raycast_im=raycast_im)
    gumbel_im = raycast_im.copy()
    gumbel_im[gumbel_mask == 1] = 0
    
    # range image -> point cloud
    if self.dataset == 'waymo':
      gumbel_pcd = unproject_range(gumbel_im, **WAYMO_RANGE_CONFIG)
      gumbel_pcd = transform_points_to_pose(gumbel_pcd, self.pose)

    elif self.dataset == 'kitti360':
      gumbel_pcd = unproject_range(gumbel_im, **KITTI_RANGE_CONFIG)
      gumbel_pcd = transform_points_to_pose(gumbel_pcd, self.pose)

    return gumbel_pcd

  def decode_hitpoints(self, hits: np.ndarray):
    '''
    given the origin, direction, and distance of each ray -> compute the 
    corresponding point cloud 
    '''
    assert len(hits) == len(self.rays) 

    mask = np.where(np.logical_and(hits != np.inf, hits < self.max_range), 1, 0)
    rays, hits = self.rays[mask == 1], hits[mask == 1]
    
    relative_hit_points = rays[:,-3:] * hits[:, np.newaxis]
    global_hit_points = rays[:,:3] + relative_hit_points

    return self._raydrop(global_hit_points)
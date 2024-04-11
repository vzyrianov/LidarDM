import numpy as np
from tqdm import tqdm

import open3d as o3d
import open3d.visualization.rendering as rendering

from lidardm.visualization import visualize_lidardm_map, render_open3d_mesh, visualize_lidar_map_aligned
from lidardm.core.datasets.utils import decode
from lidardm.lidar_generation.scene_composition.compositors import SceneCompositor

def get_visualization_videos(compositor: SceneCompositor, waymax=False, ssh=False):
  '''
  Functionalities:
  - Visualize the point cloud given a scene compositor

  Inputs:
  - compositor: the compositor itself, refers to one of the notebooks for details
  - ssh: headless rendering doesn't work via ssh, so we provide another viz 
  '''
  aligned_map_lidar = []
  aligned_mesh_lidar_bev = []
  aligned_mesh_lidar_side = []
  aligned_mesh_lidar_pts = []

  render = rendering.OffscreenRenderer(800, 800)

  for i, scene in tqdm(enumerate(compositor), total=len(compositor), desc="Grabbing viz"):

    # get map
    if waymax:
      encoded_map = compositor.get_map_idx(i)
      bev_map = visualize_lidardm_map(decode(encoded_map, 13))
    
      # align map with lidar
      final = visualize_lidar_map_aligned(bev_map, scene["scan"], scene["pose"])
      aligned_map_lidar.append(final)

    if ssh:
      bev_map = np.ones((640, 640, 3))
    
      # align map with lidar
      final = visualize_lidar_map_aligned(bev_map, scene["scan"], scene["pose"])
      aligned_map_lidar.append(final)

    else:
      meshes = []
      pcd_colors = np.tile(np.array([0.22, 0.29, 0.58]) ,(len(scene["scan"]),1))
      
      for agent_id in scene["agents"]:
        o3d_bbox = scene["agents"][agent_id]["bbox"]
        bbox = o3d.geometry.LineSet()
        bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_bbox)
        bbox.paint_uniform_color([1,0,0])
        meshes.append(bbox)

        pts_idx = o3d_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(scene["scan"]))
        pcd_colors[pts_idx] = [1,0,0]

      mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=scene["pose"][:3,3])
      mesh_frame.rotate(scene["pose"][:3,:3])
      meshes.append(mesh_frame)
        
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(scene["scan"])
      pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
      meshes.append(pcd)

      bev_img, pts_img, side_img = render_open3d_mesh(meshes, pose=scene["pose"], render=render)
      aligned_mesh_lidar_bev.append(np.array(bev_img))
      aligned_mesh_lidar_side.append(np.array(side_img))
      aligned_mesh_lidar_pts.append(np.array(pts_img))

  if ssh:
    return [aligned_map_lidar]
  
  if waymax:
    return [aligned_map_lidar, aligned_mesh_lidar_bev, aligned_mesh_lidar_side, aligned_mesh_lidar_pts]
  
  return [aligned_mesh_lidar_bev, aligned_mesh_lidar_side, aligned_mesh_lidar_pts]
  
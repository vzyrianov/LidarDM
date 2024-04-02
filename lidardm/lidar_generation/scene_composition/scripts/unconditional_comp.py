import os
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import open3d as o3d
from natsort import natsorted
from lidardm.visualization.meshify import filter_mesh
from lidardm.lidar_generation.scene_composition.compositors import GenerativeCompositor
from lidardm.visualization.open3d import render_open3d, save_open3d_render
from lidardm import PROJECT_DIR

SPATIAL_RANGE = [-50, 50, -50, 50, -3.23, 3.77]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script dump raycasted results given meshes')
  parser.add_argument('--folder', type=str, help='prototype folder')
  parser.add_argument('--n_frames', type=int, default=5)
  parser.add_argument('--n_objects', type=int, default=10)
  parser.add_argument('--raycaster', type=str, default='kitti360')
  parser.add_argument('--waymo_dataset_root', type=str, default=os.path.join(PROJECT_DIR, '_datasets', 'waymo_preprocessed'))

  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=100000)
  args = parser.parse_args()

  '''
  exampl usage: python unconditional_comp.py --folder _samples/my_kitti_uncond/
  will save into the same folder
  
  NOTE: if use meshes generated from kitti, needs to rotate the mesh like below
  '''

  out_folders = natsorted(glob(os.path.join(args.folder, "out_*")))

  for i in tqdm(range(args.start, min(args.end+1, len(out_folders)))):

    folder = out_folders[i]
    print("Processing", folder)

    mesh_file = os.path.join(folder, 'ply', 'final.ply')

    raycast_dir = os.path.join(folder, 'pcd', 'raycast')
    gumbel_dir = os.path.join(folder, 'pcd', 'gumbel')
    softmax_dir = os.path.join(folder, 'pcd', 'softmax')
    if not os.path.isdir(raycast_dir):
        os.makedirs(raycast_dir)
        os.makedirs(gumbel_dir)
        os.makedirs(softmax_dir)

    # bev_folder = os.path.join(folder, 'raycast_imgs', 'bev') 
    # side_folder = os.path.join(folder, 'raycast_imgs', 'side')
    # points_folder = os.path.join(folder, 'raycast_imgs', 'points')

    # if not os.path.isdir(bev_folder): os.makedirs(bev_folder)
    # if not os.path.isdir(side_folder): os.makedirs(side_folder)
    # if not os.path.isdir(points_folder): os.makedirs(points_folder)

    mesh = o3d.io.read_triangle_mesh(mesh_file, True)
    mesh = filter_mesh(mesh)

        
    #
    # TODO: This was commented for KITTI. Check if it is needed for Waymo. 
    #
    
    #mesh.transform(np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]))
    triangles = np.asarray(mesh.triangles)
    arr = np.vstack([triangles, np.flip(triangles, 1)])
    #mesh.triangles = o3d.utility.Vector3iVector(arr.copy())
    mesh = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(arr.copy()))
    mesh.compute_vertex_normals()

    try:
      scene_compositor = GenerativeCompositor(mesh=mesh, 
                                              waymo_dataset_root=args.waymo_dataset_root,
                                              n_frames=args.n_frames, 
                                              n_objects=args.n_objects,
                                              raycaster_config=args.raycaster)
    except KeyError:
      continue

    for i, scene in enumerate(scene_compositor):
      np.save(os.path.join(raycast_dir, f"{i}.npy"), scene["scan"]) #scene["raycast_pcd"]
      #np.save(os.path.join(gumbel_dir, f"{i}.npy"), scene["gumbel_pcd"])
      #np.save(os.path.join(softmax_dir, f"{i}.npy"), scene["softmax_pcd"])

      # bev_img, pts_img, side_img = render_open3d(scene["scans"], SPATIAL_RANGE)
      # save_open3d_render(os.path.join(bev_folder, f"{i}.png"), bev_img, quality=9)
      # save_open3d_render(os.path.join(bev_folder, f"{i}.png"), bev_img, quality=9)
      # save_open3d_render(os.path.join(bev_folder, f"{i}.png"), bev_img, quality=9)
    # break

    
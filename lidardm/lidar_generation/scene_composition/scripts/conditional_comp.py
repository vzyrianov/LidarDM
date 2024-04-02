import os
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import open3d as o3d

from lidardm.visualization.meshify import filter_mesh
from lidardm.lidar_generation.scene_composition.compositors import WaymoCompositor
from lidardm import PROJECT_DIR

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script dump raycasted results given meshes')
  parser.add_argument('--folder', type=str, help='prototype folder')
  parser.add_argument('--n_frames', type=int, default=5)
  parser.add_argument('--raycaster', type=str, default='waymo')
  parser.add_argument('--waymo_dataset_root', type=str, default=os.path.join(PROJECT_DIR, '_datasets', 'waymo_preprocessed'))
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=100000)
  args = parser.parse_args()

  '''
  exampl usage: python conditional_comp.py --folder /generated_samples
  will save into the same prototype folder
  '''

  out_folders = sorted(glob(os.path.join(args.folder, "out_*")))

  for i in tqdm(range(args.start, min(args.end+1, len(out_folders)))):

    folder = out_folders[i]

    mesh_file = os.path.join(folder, 'ply', 'final.ply')
    manifest = np.genfromtxt(os.path.join(folder, 'manifest.txt'), delimiter=", ", dtype='str')
    seq_id, center_frame_idx = manifest[0], int(manifest[1])

    output_dir = os.path.join(folder, 'raycast')
    pose_path = os.path.join(folder, 'poses.txt')
    # if os.path.isdir(output_dir): continue
    # os.makedirs(output_dir)

    #if not os.path.isdir(output_dir): 
    #  os.makedirs(output_dir)

    try:
      os.makedirs(output_dir)
    except:
      continue

    mesh = o3d.io.read_triangle_mesh(mesh_file, True)
    #mesh.transform(np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]))
    triangles = np.asarray(mesh.triangles)
    arr = np.vstack([triangles, np.flip(triangles, 1)])
    #mesh.triangles = o3d.utility.Vector3iVector(arr.copy())
    mesh = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(arr.copy()))
    mesh.compute_vertex_normals()
    mesh = filter_mesh(mesh)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene_compositor = WaymoCompositor(sequence_id=seq_id, 
                                       anchor_frame_idx=center_frame_idx,
                                       mesh=mesh_t,
                                       n_frames=args.n_frames,
                                       waymo_dataset_root=args.waymo_dataset_root,)

    poses = []
    for i, scene in enumerate(scene_compositor):
      np.save(os.path.join(output_dir, f"{i}.npy"), scene["scan"])
      poses.append(scene["pose"].flatten())

    np.vstack(poses)
    np.savetxt(pose_path, poses)
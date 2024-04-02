import open3d as o3d
import matplotlib.cm as cm
import os
from tqdm import tqdm

import argparse
import numpy as np
from natsort import natsorted 

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from kitti360scripts.helpers.annotation import Annotation3D

trans_velo_to_imu = np.array(
        [[0.99992906, 0.0057743, 0.01041756, 0.77104934],
         [0.00580536, -0.99997879, -0.00295331, 0.29854144],
         [0.01040029, 0.00301357, -0.99994137, -0.83628022],
         [0, 0, 0, 1]])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script that preprocess KITTI360 for dynamic filtering and pose interpolation')
  parser.add_argument('--seq', type=int, nargs='+', help='sequence', default=0)
  parser.add_argument('--dataset_path', type=str, help='path to kitti360 dataset')
  parser.add_argument('--out_dir', type=str, help='output folder')
  parser.add_argument('--viz', action=argparse.BooleanOptionalAction, help='visualize or nah')
  args = parser.parse_args()

  os.environ["KITTI360_DATASET"] = args.dataset_path
  
  for seq in args.seq:
    print("Processing Sequence", seq)
    
    sequence_name = '2013_05_28_drive_%04d_sync' % seq
    
    bbox_file = os.path.join(args.dataset_path, 'data_3d_bboxes')
    annotation3D = Annotation3D(bbox_file, sequence_name)

    pose_file = os.path.join(args.dataset_path, 'data_poses', sequence_name, 'poses.txt')
    poses = np.loadtxt(pose_file, delimiter=' ', dtype=float)
    pose_frames, poses = poses[:,0], poses[:,1:]

    scan_folder = os.path.join(args.dataset_path, 'data_3d_raw', sequence_name, 'velodyne_points', 'data')
    pc_filenames = natsorted(os.listdir(scan_folder)) 

    output_folder = os.path.join(args.output_dir, 'data_3d_static', sequence_name, 'velodyne_points', 'data')
    output_pose_file = os.path.join(args.output_dir, 'data_interp_poses', sequence_name, 'poses.txt')

    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
    print("Saving scans to", output_folder)

    if args.viz:
      vis = o3d.visualization.Visualizer()
      vis.create_window()

    lerped_poses = []
    
    for i in tqdm(range(len(pc_filenames))):
      output_file = os.path.join(output_folder, pc_filenames[i])
      scan_frames = int(pc_filenames[i][:-4]) # -4 to cut off ".bin"
      
      if args.viz:
        vis.clear_geometries()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0,0,0])
        vis.add_geometry(mesh_frame)

      current_lidar_scan = os.path.join(scan_folder, pc_filenames[i])
      points = np.fromfile(current_lidar_scan, dtype=np.float32).reshape((-1, 4))[:, :3]
      
      # get the latest pose without passing the frame idx
      # pose is Tr(world -> current_lidar_frame)
      pose_idx = int(np.searchsorted(pose_frames, scan_frames))

      # no reliable first pose
      if scan_frames < pose_frames[0]:
        continue

      # runs out of pose
      if pose_idx == poses.shape[0] or pose_frames[pose_idx] < scan_frames:
        break

      # current pose
      if pose_frames[pose_idx] == scan_frames:
        pose = poses[pose_idx,:].reshape(3,-1)
        pose = np.vstack((pose, [0,0,0,1])) 

      # interpolate between poses 
      elif pose_frames[pose_idx] > scan_frames:
        pose_after = poses[pose_idx,:].reshape(3,-1)
        pose_before = poses[pose_idx-1,:].reshape(3,-1)

        # rotation slerp
        key_rots = R.from_matrix([pose_before[:3,:3], pose_after[:3,:3]])
        key_times = [pose_frames[pose_idx-1], pose_frames[pose_idx]]

        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(scan_frames).as_matrix()

        pose = np.eye(4)
        pose[:3,:3] = interp_rots

        # translation lerp
        alpha = 0
        if pose_frames[pose_idx-1] != pose_frames[pose_idx]:
          alpha = (scan_frames - pose_frames[pose_idx-1]) / (pose_frames[pose_idx] - pose_frames[pose_idx-1])

        pose[:3,3] = (1.0 - alpha) * pose_before[:3,3] + alpha * pose_after[:3,3]

      lerped_poses.append(np.append(scan_frames, pose.flatten()))
      pose = np.linalg.inv(pose @ trans_velo_to_imu) 

      for globalId,v in annotation3D.objects.items():
        for obj in v.values():
          if obj.timestamp != -1 and obj.timestamp >= i-1 and obj.timestamp <= i+1:
            # create a 3D bbox mesh
            bbox_mesh = o3d.geometry.TriangleMesh()
            bbox_mesh.vertices = o3d.utility.Vector3dVector(obj.vertices)
            bbox_mesh.triangles = o3d.utility.Vector3iVector(obj.faces)
            bbox_mesh.transform(pose)

            bbox_mesh.scale(1.5, bbox_mesh.get_center())

            # delete all points inside the bounding box 
            bbox = bbox_mesh.get_oriented_bounding_box()
            mask = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
            points = np.delete(points, mask, axis=0)

            if args.viz:
              bbox = o3d.geometry.LineSet.create_from_triangle_mesh(bbox_mesh)
              vis.add_geometry(bbox)

      # save preprocessed points
      points.tofile(os.path.join(output_folder, pc_filenames[i]))

      if args.viz:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(pcd)
        vis.poll_events()

    # save interpolated poses
    lerped_poses = np.vstack(lerped_poses)
    np.savetxt(output_pose_file, lerped_poses, fmt=' '.join(['%i'] + ['%1.4f']*16))

    if args.viz:
      vis.destroy_window()
      
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import nksr
import torch
import argparse
import os
import gzip
import copy

from natsort import natsorted 
from pycg import vis
import open3d as o3d
import numpy as np
from common_field_generation import *
from tqdm import tqdm

LIDAR_HEIGHT = 2.18400000e+00
MOVEMENT_THRESHOLD = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script that takes two inputs of different types.')
    parser.add_argument('--scans', type=str, help='point cloud folder')
    parser.add_argument('--pose', type=str, help='pose file')
    parser.add_argument('--splits', type=str, nargs='+', help='splits')
    parser.add_argument('--acc', type=int, default=15)
    parser.add_argument('--start_scan', type=int, default=0)
    parser.add_argument('--end_scan', type=int, default=100000)
    parser.add_argument('--out_dir', type=str, help="out directory")

    parser.add_argument('--voxel_size', type=float, default=0.15)
    parser.add_argument('--grid_size', type=int, nargs='+')
    parser.add_argument('--save_files', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    warning_on_low_memory(20000.0)

    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")
    reconstructor.hparams.voxel_size = args.voxel_size

    dataset_out_dir = args.out_dir

    for split_type in args.splits:
        print("Processing", split_type)
        tfrecords_folder = os.path.join(args.scans, split_type)

        tfrecords = natsorted(os.listdir(tfrecords_folder)) 
        for tfrecord_idx in tqdm(range(args.start_scan, min(args.end_scan+1, len(tfrecords)))):
            tfrecord = tfrecords[tfrecord_idx]
            tfrecord_output_folder = os.path.join(dataset_out_dir, split_type, tfrecord)
   
            # skip if exists
            if os.path.isdir(tfrecord_output_folder):
                continue

            # create output folders
            grid_folder = os.path.join(dataset_out_dir, split_type, tfrecord, 'grid')
            mesh_folder = os.path.join(dataset_out_dir, split_type, tfrecord, 'mesh')
            os.makedirs(grid_folder)
            os.makedirs(mesh_folder)
   

            accumulate = args.acc
            pc_folder = os.path.join(args.scans, split_type, tfrecord, "lidar_scans")
            pose_file = os.path.join(args.pose, split_type, tfrecord, "poses.txt")

            pc_filenames = natsorted(os.listdir(pc_folder)) 
            frame_count = len(pc_filenames)

            frames, poses = read_poses_file(pose_file)
            assert len(frames) == frame_count, f'unevent pose vs points size {len(frames)} vs {frame_count}'

            def get_pose(idx: int):
                pose = poses[idx,:].reshape(4,4)
                return pose

            points_cache = []
            for i in tqdm(range(0, frame_count - accumulate)):

                # break if not enough frames for a full accumulated pcd
                if i + accumulate > frame_count:
                    break

                # break if insufficient movement
                first_pose = get_pose(i)
                last_pose = get_pose(min(i + accumulate - 1, frame_count - 1))
                dist = np.linalg.norm(first_pose[:2, 3] - last_pose[:2, 3])
                if dist < MOVEMENT_THRESHOLD:
                    continue

                # get start and end frame for naming purposes
                first_frame = int(pc_filenames[i][:-4])
                last_frame = int(pc_filenames[min(i + accumulate - 1, frame_count - 1)][:-4])
                current_frame_range = [first_frame, last_frame]

                # filenames
                grid_file = os.path.join(grid_folder, f'{current_frame_range[0]}-{current_frame_range[1]}.npy.gz')

                # continue if exists
                if os.path.isfile(grid_file):
                    continue

                # populate the points cache if empty
                if len(points_cache) == 0:
                    for k in range(i, i + accumulate, 1):
                        frame_filename = os.path.join(pc_folder, pc_filenames[k])
                        frame_pcd = read_point_cloud(frame_filename)
                        points_cache.append(frame_pcd)

                # otherwise slide the window over by one
                else:
                    frame_filename = os.path.join(pc_folder, pc_filenames[i + accumulate - 1])
                    frame_pcd = read_point_cloud(frame_filename)
                    points_cache.append(frame_pcd)
                    points_cache.pop(0)

                # center pose is pose at the middle of the batch, whose location is considered 0,0,0
                center_pose = get_pose(i+((accumulate-1)//2))
                center_pose = np.linalg.inv(center_pose)

                # accumulated pcd for this batch, store sensor position as colors
                batch_pcd = o3d.geometry.PointCloud()
                for j in range(accumulate):
                    frame_pcd = copy.deepcopy(points_cache[j])

                    pose = get_pose(i+j)
                    pose = center_pose @ pose
                    frame_pcd = frame_pcd.transform(pose)
        
                    # vehicle frame defined as lidar frame with z = 0
                    sensor_position = pose[:3, 3] + [0,0,LIDAR_HEIGHT]
                    sensor_position_np = np.tile(sensor_position, (len(frame_pcd.points), 1))
                    frame_pcd.colors = o3d.utility.Vector3dVector(sensor_position_np)

                    batch_pcd += frame_pcd

                input_xyz = torch.from_numpy(np.asarray(batch_pcd.points)).float().to(device)
                input_sensor = torch.from_numpy(np.asarray(batch_pcd.colors)).float().to(device)
                
                dmc_vertices, dmc_graph = build_grid(args.voxel_size, np.array(args.grid_size))
                dmc_vertices, dmc_value, dmc_graph = generate_dmc(reconstructor, input_xyz, input_sensor, dmc_vertices, dmc_graph, args.voxel_size)
                
                # save to file
                if args.save_files:

                    if i % (frame_count / 100) == 0:
                        dual_v, dual_f = get_mesh(dmc_vertices, dmc_value, dmc_graph)
                        vis.to_file(vis.mesh(dual_v, dual_f), os.path.join(mesh_folder, f'{current_frame_range[0]}-{current_frame_range[1]}.ply'))
                    
                    output_grid = torch.hstack((dmc_vertices, dmc_value[:,None]))
                    output_grid = output_grid.to("cpu").numpy()

                    with gzip.GzipFile(grid_file, "w") as fd:
                        np.save(fd, output_grid)
                
                    

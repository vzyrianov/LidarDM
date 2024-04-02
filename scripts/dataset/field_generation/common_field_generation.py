# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import sys
import numpy as np
import open3d as o3d

from pycg import exp
import torch 

import nksr
from nksr.ext import _CpuIndexGrid, _CudaIndexGrid
from nksr.meshing import MarchingCubes
from nksr.ext import meshing

DOWNLOAD_URL = "https://nksr.huangjh.tech"

device = torch.device("cuda:0")

def build_vert_and_graph(voxel_size: float, dense_grid: torch.Tensor):
    # cuda index grid from nksr
    dense_cuda_grid = _CudaIndexGrid(voxel_size, 0, device.index)
    dense_cuda_grid.build_from_pointcloud(dense_grid, (0, 0, 0), (0, 0, 0))
    dense_cuda_grid = [dense_cuda_grid]

    dual_grid = meshing.build_joint_dual_grid(dense_cuda_grid)
    dmc_graph = meshing.dual_cube_graph(dense_cuda_grid, dual_grid)
    
    # build new vertices
    dmc_vertices = torch.cat([
        f_grid.grid_to_world(f_grid.active_grid_coords().float())
        for f_grid in dense_cuda_grid if f_grid.num_voxels() > 0
    ], dim=0)

    # print("[DENSE_GRID]",dense_grid[0])
    # print("[VERTICES]",dmc_vertices[0])

    assert dense_grid.size() == dmc_vertices.size(), f'Grid differs in size! {dense_grid.size()} vs {dmc_vertices.size()}'
    del dense_cuda_grid, dual_grid, dense_grid

    return dmc_vertices, dmc_graph

def build_grid(voxel_size: float, grid_size: np.ndarray):
    meshgrid_x = np.arange(0 - grid_size[0] / 2.0, 0 + grid_size[0] / 2.0, voxel_size)
    meshgrid_y = np.arange(0 - grid_size[1] / 2.0, 0 + grid_size[1] / 2.0, voxel_size)
    meshgrid_z = np.arange(0 - grid_size[2] / 2.0, 0 + grid_size[2] / 2.0, voxel_size)
    # print("[SIZE", meshgrid_x.shape, meshgrid_y.shape, meshgrid_z.shape)
    # generate dense voxel grid
    dense_grid = np.vstack(np.meshgrid(meshgrid_x,meshgrid_y,meshgrid_z)).reshape(3,-1).T
    dense_grid = torch.from_numpy(dense_grid).float().to(device)
    
    # cuda index grid from nksr
    dmc_vertices, dmc_graph = build_vert_and_graph(voxel_size, dense_grid)

    return dmc_vertices, dmc_graph

def generate_dmc(reconstructor: nksr.Reconstructor, 
                 input_xyz: torch.Tensor, 
                 input_sensor: torch.Tensor,
                 dmc_vertices: torch.Tensor,
                 dmc_graph: torch.Tensor,
                 voxel_size: float):
    
    field = reconstructor.reconstruct(
        input_xyz, sensor=input_sensor, detail_level=None,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        # Chunked reconstruction (if OOM)
        # chunk_size=51.2,
        preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0)
    )

    dmc_value = field.evaluate_f_bar(dmc_vertices, max_points=-1)
    
    # rough mesh
    dmc_mask = field.mask_field.evaluate_f_bar(dmc_vertices, max_points=-1) < 0.0

    dmc_vertices_new = dmc_vertices[dmc_mask==1]
    dmc_value_new = dmc_value[dmc_mask==1]

    dmc_vertices_new, dmc_graph_new = build_vert_and_graph(voxel_size, dmc_vertices_new)

    return dmc_vertices_new, dmc_value_new, dmc_graph_new

def get_mesh(dmc_vertices, dmc_value, dmc_graph):
    dual_v, dual_f = meshify(dmc_vertices, dmc_value, dmc_graph)
    return dual_v, dual_f

def meshify(dmc_vertices: torch.Tensor, 
            dmc_value: torch.Tensor, 
            dmc_graph: torch.Tensor):

    dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_value)

    return dual_v, dual_f

def warning_on_low_memory(threshold_mb: float):
    gpu_status = exp.get_gpu_status('localhost')
    if len(gpu_status) == 0:
        exp.logger.fatal("No GPU found!")
        return

    gpu_status = gpu_status[0]
    available_mb = (gpu_status.gpu_mem_total - gpu_status.gpu_mem_byte) / 1024. / 1024.

    if available_mb < threshold_mb:
        exp.logger.warning("Available GPU memory is {:.2f} MB, "
                           "we recommend you to have more than {:.2f} MB available.".format(available_mb, threshold_mb))


def read_poses_file(filename):
    poses = np.loadtxt(filename, delimiter=' ', dtype=float)
    frames, poses = poses[:,0], poses[:,1:]

    return frames, poses

def read_point_cloud(filename: str):
    if ".bin" in filename:
        points = np.fromfile(filename, dtype=np.float32).reshape((-1, 3))[:, :3]

    elif ".ply" in filename or ".pcd" in filename:
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points)

    elif ".npy" in filename:
        points = np.load(filename)

    else:
        sys.exit(f'Unsupported format {filename}')

    min_z = -3.0
    min_range = 3.0 
    max_range = 50.0

    preprocessed_points = preprocess(points, min_z, min_range, max_range)
    pc_out = o3d.geometry.PointCloud()
    pc_out.points = o3d.utility.Vector3dVector(preprocessed_points)

    return pc_out

def preprocess(points, z_th, min_range, max_range):
    dist = np.linalg.norm(points, axis=1)
    filtered_idx = (dist > min_range) & (dist < max_range) & (points[:, 2] > z_th)
    return points[filtered_idx]

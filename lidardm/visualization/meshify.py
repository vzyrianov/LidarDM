

import torch
import copy
from pycg import vis
import open3d as o3d
import numpy as np

import os
from typing import Tuple

from nksr.ext import _CudaIndexGrid
from nksr.meshing import MarchingCubes
from nksr.ext import meshing

def filter_mesh(mesh):
  triangle_clusters, cluster_n_triangles, cluster_area = (
    mesh.cluster_connected_triangles())
  triangle_clusters = np.asarray(triangle_clusters)
  cluster_n_triangles = np.asarray(cluster_n_triangles)
  cluster_area = np.asarray(cluster_area)

  largest_cluster_idx = cluster_n_triangles.argmax()
  triangles_to_remove = triangle_clusters != largest_cluster_idx
  mesh.remove_triangles_by_mask(triangles_to_remove)
  return mesh

class FieldsMeshifier():
  '''
  Usage:
    meshifier = FieldsMeshifier(device, 0.15, [96, 96, 6])
    mesh = meshifier(field) # output mesh is o3d triangle mesh in GPU
  '''
  def __init__(
      self, 
      device: torch.device, 
      voxel_size: float, 
      grid_size: np.ndarray,
      use_post_process=True,
    ) -> None:
    
    self.device = device
    self.voxel_size = voxel_size
    self.grid_size = np.array(grid_size)
    self.use_post_process = use_post_process

    self._build_vertices_and_graph()
    self._get_grid_order()

    self.rotation = np.array([[0, 1,  0,  0],
                            [1, 0,  0,  0],
                            [0, 0,  1,  0],
                            [0, 0,  0,  1.0]])

  def _build_vertices_and_graph(self) -> torch.Tensor:
    '''
    inputs:
      voxel_size: distance between adjacent voxel corners in meters 
      grid_size: [l w h] in ijk coordinates

      example: voxel_size 0.15m and grid_size [96 96 6] will generate a 640x640x40 grid (in meters) 

    outputs:
      grid centered at 0,0,0 in Cartesian coord
    '''

    # create a uniform lwh grid with uniform grid_size
    meshgrid_x = np.arange(0 - self.grid_size[0] / 2.0, 0 + self.grid_size[0] / 2.0, self.voxel_size)
    meshgrid_y = np.arange(0 - self.grid_size[1] / 2.0, 0 + self.grid_size[1] / 2.0, self.voxel_size)
    meshgrid_z = np.arange(0 - self.grid_size[2] / 2.0, 0 + self.grid_size[2] / 2.0, self.voxel_size)

    dense_grid = np.vstack(np.meshgrid(meshgrid_x,meshgrid_y,meshgrid_z)).reshape(3,-1).T
    dense_grid = torch.from_numpy(dense_grid).float().to(self.device)
    
    # cuda index grid from nksr
    dense_cuda_grid = _CudaIndexGrid(self.voxel_size, 0, self.device.index)
    dense_cuda_grid.build_from_pointcloud(dense_grid, (0, 0, 0), (0, 0, 0))
    dense_cuda_grid = [dense_cuda_grid]

    # build dmc prerequisites from nksr
    dual_grid = meshing.build_joint_dual_grid(dense_cuda_grid)
    dmc_graph = meshing.dual_cube_graph(dense_cuda_grid, dual_grid)
    
    # build dmc vertices from the dense grid 
    dmc_vertices = torch.cat([
      f_grid.grid_to_world(f_grid.active_grid_coords().float())
      for f_grid in dense_cuda_grid if f_grid.num_voxels() > 0
    ], dim=0)

    assert dense_grid.size() == dmc_vertices.size(), f'Grid differs in size! {dense_grid.size()} vs {dmc_vertices.size()}'
    del dense_cuda_grid, dual_grid, dense_grid

    self.dmc_vertices = dmc_vertices
    self.dmc_graph = dmc_graph

  def _get_grid_order(self) -> torch.Tensor:
    '''
    return an index tensor grid_order that redo sorting
    example: A == sort(B) -> B[grid_order] == A
    '''

    grid_indices = self._cartesian_to_indices(self.dmc_vertices)
    grid_order = torch.argsort(torch.argsort(grid_indices))

    self.grid_order = grid_order
  
  def _cartesian_to_indices(self, input: torch.Tensor) -> torch.Tensor:
    '''
    input: Nx3 tensor that represents coordinates of points in Cartesian coordinates
    outputs: Nx1 tensor that represents the row-major indices
    '''

    cartesian_grid_size = self.grid_size / self.voxel_size
    indices = input / self.voxel_size

    indices[:,0] += cartesian_grid_size[0] / 2
    indices[:,1] += cartesian_grid_size[1] / 2
    indices[:,2] += cartesian_grid_size[2] / 2

    indices = torch.round(indices).int()
    
    return (indices[:, 0] * cartesian_grid_size[1] * cartesian_grid_size[2] + 
        indices[:, 1] * cartesian_grid_size[2] + 
        indices[:, 2]).int()
     
  def _meshify(
      self, 
      dmc_vertices: torch.Tensor, 
      dmc_value: torch.Tensor, 
      dmc_graph: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    '''
    apply dual marching cube on the sdf field to generate mesh
    '''

    dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_value)

    return dual_v, dual_f
  
  def post_process(self, current_mesh):
    current_mesh.transform(self.rotation)
      
    triangles = np.asarray(current_mesh.triangles)
    arr = np.vstack([triangles, np.flip(triangles, 1)])
    current_mesh = o3d.geometry.TriangleMesh(current_mesh.vertices, o3d.utility.Vector3iVector(arr.copy()))
    current_mesh.compute_vertex_normals()
    return current_mesh

  def generate_mesh(
      self,
      input_field: torch.Tensor,
      mesh_file = None,
    ) -> o3d.geometry.TriangleMesh:

    '''
    input: input_fields expected to be Nx4 (x y z occupancy_val)
    output: mesh
    '''
    coords, values = input_field[:,:3], input_field[:,3]
    indices = self._cartesian_to_indices(coords)

    dmc_vertices = copy.deepcopy(self.dmc_vertices)
    dmc_graph = copy.deepcopy(self.dmc_graph)
    dmc_values = torch.ones_like(dmc_vertices[:,0]) * -100

    dmc_values[indices] = values
    dmc_values = dmc_values[self.grid_order]

    dual_v, dual_f = self._meshify(dmc_vertices, dmc_values, dmc_graph)
    mesh = vis.mesh(dual_v, dual_f)

    mesh = filter_mesh(mesh)
    if self.use_post_process:
      mesh = self.post_process(mesh)
    
    if(mesh_file is not None):
      vis.to_file(mesh, os.path.join(mesh_file))

    return mesh

  def get_buffer_pcd(self) -> o3d.geometry.PointCloud:
    '''
    draw_plotly stretch the mesh vertically, so add 8 points on the corners
    to counter the effect
    '''
    coord = np.array([-50, 50])
    points = np.vstack(np.meshgrid(coord,coord,coord)).reshape(3,-1).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd
    
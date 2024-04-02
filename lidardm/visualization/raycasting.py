import numpy as np
from lidardm.lidar_generation.scene_composition.utils.raycast import *
import open3d as o3d


__all__ = ["raycast_kitti_util"]

def raycast_kitti_util(mesh):
    mesh.compute_vertex_normals()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    raycaster = Raycaster(**KITTI_RAYCAST_LARGE_CONFIG)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    rays = raycaster.generate_rays(pose=np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        ]))
    ans = scene.cast_rays(rays)
    lidar_points = raycaster.decode_hitpoints(ans['t_hit'].numpy())

    return lidar_points
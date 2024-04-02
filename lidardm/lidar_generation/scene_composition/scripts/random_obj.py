import copy

import open3d as o3d
import argparse
import numpy as np
from PIL import Image

from agents.agent import Vehicle, Pedestrian
from raycast import Raycaster, KITTI_RAYCAST_CONFIG

def create_bbox(height, width, length):
  bbox = o3d.geometry.LineSet()
  points = np.vstack(np.meshgrid(length,width,height)).reshape(3,-1).T
  bbox.points = o3d.utility.Vector3dVector(points)
  lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7]])
  bbox.lines = o3d.utility.Vector2iVector(lines)

  bbox = bbox.get_oriented_bounding_box()
  return bbox

def create_agent_db():
  car_1 = Vehicle(original_pose=np.array([-2, 0, 0]), speed=1.05)
  car_2 = Vehicle(original_pose=np.array([-2.5, -12, 0]), speed=0.99)
  car_3 = Vehicle(original_pose=np.array([-2, -25, 0]), speed=0.98)
  car_4 = Vehicle(original_pose=np.array([2, 7, 0]) , speed=0)
  car_5 = Vehicle(original_pose=np.array([3, 24, 0]), speed=0)
  car_6 = Vehicle(original_pose=np.array([2, 12, 0]), speed=0)
  car_7 = Vehicle(original_pose=np.array([-1.5, -30, 0]), speed=0.97)
  car_8 = Vehicle(original_pose=np.array([-2.5, -35, 0]), speed=0.93)

  ped_1 = Pedestrian(original_pose=np.array([-2.5, -15, 0]), speed=0.2)
  ped_2 = Pedestrian(original_pose=np.array([6, 0, 0]), speed=0.2)
  ped_3 = Pedestrian(original_pose=np.array([8, 0, 0]), speed=0.2)
  ped_4 = Pedestrian(original_pose=np.array([5, 17, 0]), speed=-0.2)
  ped_5 = Pedestrian(original_pose=np.array([7, 1, 0]), speed=0.2)
  ped_6 = Pedestrian(original_pose=np.array([8, -18, 0]), speed=0.2)
  ped_7 = Pedestrian(original_pose=np.array([5.5, 15, 0]), speed=0.2)

  car_agents = [car_1, car_2, car_3, car_4, car_5, car_6, car_7, car_8]
  ped_agents = [ped_1, ped_2, ped_3, ped_4, ped_5, ped_6, ped_7]

  return car_agents, ped_agents

def main(args):
  vis = o3d.visualization.Visualizer()
  vis.create_window()

  car_agents, ped_agents = create_agent_db()

  pedestrian_bbox = create_bbox(np.array([-1,0.5]), np.array([0,1]), np.array([0,1]))
  car_bbox = create_bbox(np.array([-1,1]), np.array([0,4]), np.array([0,2.5]))

  nksr_mesh = o3d.io.read_triangle_mesh(args.mesh)
  nksr_mesh = nksr_mesh.filter_smooth_laplacian(lambda_filter=1, number_of_iterations=2)
  nksr_mesh.compute_vertex_normals()
  vis.add_geometry(nksr_mesh)

  poses = np.zeros((30,3))
  poses[:,1] = np.arange(-20, 10, 1)
  pcd = o3d.geometry.PointCloud()

  meshes = []
  for i in range(len(poses)):
    pose = poses[i]
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=pose)
    vis.add_geometry(mesh_frame)
    meshes.append(mesh_frame)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(nksr_mesh))
    vlp64 = Raycaster(**KITTI_RAYCAST_CONFIG)

    for car in car_agents:
      car_bbox_copy = copy.deepcopy(car_bbox)
      car_bbox_copy.translate(car.original_pose + [0,car.speed * i,0])
      vis.add_geometry(car_bbox_copy)
      meshes.append(car_bbox_copy)

      car_mesh = car.get_mesh_at_bbox(bbox=car_bbox_copy, frame=i)
      vis.add_geometry(car_mesh)
      meshes.append(car_mesh)
      scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(car_mesh))

    for ped in ped_agents:
      ped_bbox_copy = copy.deepcopy(pedestrian_bbox)
      ped_bbox_copy.translate(ped.original_pose + [0,ped.speed * i,0])
      vis.add_geometry(ped_bbox_copy)
      meshes.append(ped_bbox_copy)

      ped_mesh = ped.get_mesh_at_bbox(bbox=ped_bbox_copy, frame=i)
      vis.add_geometry(ped_mesh)
      meshes.append(ped_mesh)
      scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ped_mesh))

    rays = vlp64.generate_rays(pose)
    ans = scene.cast_rays(rays)
      
    lidar_points = vlp64.decode_hitpoints(ans['t_hit'].numpy())
    
    pcd.points = o3d.utility.Vector3dVector(lidar_points)

    if i == 0:
      vis.add_geometry(pcd)
    else:
      vis.update_geometry(pcd)

    view_ctl = vis.get_view_control()
    view_ctl.set_up([0,0,1])
    view_ctl.set_front([0,-1,0.7])
    view_ctl.set_lookat(pose + [0,-5,5])
    view_ctl.set_zoom(0.005)

    if args.save_frames:
      buf = vis.capture_screen_float_buffer(do_render=True)
      buf = np.asarray(buf)
      buf = (buf * 255.).astype('uint8')
      im = Image.fromarray(buf)
      im.save(f'scenarios/pcd_only/img_{i}.png')

    vis.run()
    for mesh in meshes:
      vis.remove_geometry(mesh)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='A script that takes two inputs of different types.')
  parser.add_argument('--mesh', type=str, help='mesh file')
  parser.add_argument('--pose', type=str, help='pose file')
  parser.add_argument('--scans', type=str, help='point cloud folder')
  parser.add_argument('--save_frames', action=argparse.BooleanOptionalAction)
  
  parser.add_argument('--mode', type=str, help="ply/npy")
  args = parser.parse_args()
  main(args)
import os
import random
import open3d as o3d
import numpy as np
from natsort import natsorted
from pathlib import Path
from glob import glob

from lidardm import PROJECT_DIR

VEHICLE_MESH_FOLDER = 'generated_assets/vehicle'
PEDESTRIAN_MESH_FOLDER = 'generated_assets/pedestrian'

__all__ = ["Vehicle", "Pedestrian"]

class Agent:
  '''
  
  '''
  def __init__(self, mesh_location, original_pose=[0,0,0], speed=0, is_animated=False) -> None:
    self.is_animated = is_animated
    self.mesh_location = mesh_location
    
    if is_animated: assert os.path.isdir(mesh_location), "Expected mesh folder for animated agents"
    else: assert os.path.isfile(mesh_location), "Expected mesh file for unanimated agents"
  
    self.previous_box_center = None
    self.previous_yaw = None

    self.original_pose = original_pose
    self.speed = speed

    self.random_seed = np.random.randint(10)

  def estimate_orientation_from_bbox(self, bbox, mesh):
    yaw=np.arctan2(bbox.R[1,0],bbox.R[0,0])
    rotation_matrix = [[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]
    mesh.rotate(rotation_matrix)
    self.previous_yaw = yaw
    return mesh
  
  def estimate_orientation_from_prev_center(self, bbox, mesh):
    headings = bbox.get_center() * [1,1,0] - self.previous_box_center
    if(np.linalg.norm(headings) < 0.1):
      return self.estimate_orientation_from_bbox(bbox, mesh)
    else: 
      # yaw = np.arccos(np.dot(headings / np.linalg.norm(headings), [1,0,0]))
      yaw = np.arctan2(headings[1], headings[0])
    rotation_matrix = [[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]
    mesh.rotate(rotation_matrix)
    self.previous_yaw = yaw
    return mesh
  
  def apply_yaw_rotation(self, mesh, yaw):
    rotation_matrix = [[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]
    mesh.rotate(rotation_matrix)
    self.previous_yaw = yaw
    return mesh

  def get_mesh_at_bbox(self, frame, bbox, yaw=None):
    mesh_file = self.mesh_location

    if os.path.isdir(mesh_file):
      mesh_list = natsorted(glob(os.path.join(mesh_file, "*.obj")))
      mesh_file = os.path.join(mesh_file, mesh_list[(self.random_seed + frame + 3) % len(mesh_list)])

    agent_mesh = o3d.io.read_triangle_mesh(mesh_file, True)
    agent_mesh.compute_vertex_normals()

    # rotate so that meshes are upright
    agent_mesh.rotate([[1,0,0],[0,0,-1],[0,1,0]], [0,0,0])

    # pedestrian needs to rotate again
    if self.is_animated:
      agent_mesh.rotate(np.array([[0,-1,0],[1,0,0],[0,0,1]]), [0,0,0])

    # scale the mesh to fit the bounding box
    bbox_max_bound = bbox.get_max_bound()[2] - bbox.get_min_bound()[2]
    agent_max_bound = agent_mesh.get_max_bound()[2] - agent_mesh.get_min_bound()[2]
    agent_mesh.scale(0.8 * bbox_max_bound / agent_max_bound, agent_mesh.get_center())
    
    # rotate the mesh to the direction of the bounding box
    if yaw is not None: agent_mesh = self.apply_yaw_rotation(agent_mesh, yaw)
    elif self.previous_box_center is not None: agent_mesh = self.estimate_orientation_from_prev_center(bbox, agent_mesh)
    else: agent_mesh = self.estimate_orientation_from_bbox(bbox, agent_mesh)

    self.previous_box_center = bbox.get_center() * [1,1,0]

    # translate the mesh to the bbox location 
    agent_mesh.translate(bbox.get_center(), False)
    agent_mesh.translate([0,0,bbox.get_min_bound()[2] - agent_mesh.get_min_bound()[2]], True)

    return agent_mesh

class Vehicle(Agent):
  def __init__(self, original_pose=[0,0,0], speed=0, mesh_location=None) -> None:
    if mesh_location is None:
      vehicle_mesh_files = natsorted(glob(os.path.join(PROJECT_DIR, VEHICLE_MESH_FOLDER, '*.obj')))
      random_vehicle_file = random.choice(vehicle_mesh_files)
      mesh_location = random_vehicle_file
    super().__init__(mesh_location=mesh_location, original_pose=original_pose, speed=speed, is_animated=False)
  
class Pedestrian(Agent):
  def __init__(self, original_pose=[0,0,0], speed=0, mesh_location=None) -> None:
    if mesh_location is None:
      
      ped_animated_folders = glob(os.path.join(PROJECT_DIR, PEDESTRIAN_MESH_FOLDER, "*", ""), recursive = True) 
      random_ped_folder = random.choice(ped_animated_folders)
      mesh_location = random_ped_folder
    super().__init__(mesh_location=mesh_location, original_pose=original_pose, speed=speed, is_animated=True)

import numpy as np
from tqdm import tqdm

from waymax import datatypes

from lidardm import PROJECT_DIR
from lidardm.lidar_generation.scene_composition.utils.raycast import *
from lidardm.lidar_generation.scene_composition.compositors.trajectory_sampler import *
from lidardm.lidar_generation.scene_composition.compositors.scene_compositor import SceneCompositor
from lidardm.core.datasets.utils import encode, decode
from lidardm.core.services.sampling import *

from .waymax_utils import *
from .waymax_map import *

class WaymaxCompositor(SceneCompositor):
  def __init__(self,
               scenario: datatypes.SimulatorState,
               use_log_traj: bool = True,
               mesh: o3d.geometry.TriangleMesh = None,
               start: int = 0,
               steps: int = -1,
               raycaster_config: str = 'waymo') -> None:
    '''
    Waymax Compositor always treat first frame as center pose for simplicity
    '''

    # parameters
    self.scenario = scenario
    self.use_log_traj = use_log_traj
    self.traj = self.scenario.log_trajectory if use_log_traj else self.scenario.sim_trajectory
    self.start = start
    self.steps = steps

    # maps
    self.maps = self.create_map_sequence()
    
    # scene compositor parameters
    self.ego_poses = self.get_ego_poses(scenario, use_log_traj, start, steps)
    agents = self.create_agent_dict()

    if mesh is None:
      # mesh sampler
      self.model = instantiate_conditional_model(vae_path=os.path.join(PROJECT_DIR, "pretrained_models", "waymo", "scene_gen", "wfsvae_kl1e-7.ckpt"),
                                                  map_vae_path=os.path.join(PROJECT_DIR, "pretrained_models", "waymo", "scene_gen", "mvae.ckpt"),
                                                  unet_path=os.path.join(PROJECT_DIR, "pretrained_models", "waymo", "scene_gen", "wfsunetc.ckpt"))
      self.mesh = self.generate_mesh()
    else:
      self.mesh = mesh

    super().__init__(mesh=self.mesh, raycaster_config=raycaster_config)
    super().update_scenarios(self.ego_poses, agents)

  def __len__(self):
    return len(self.ego_poses)
  
  def __getitem__(self, idx):	
    return super().__getitem__(frame=idx)
  
  def get_map_idx(self, idx):
    return self.maps[idx].copy()

  def create_agent_dict(self) -> AgentDict:
    obj_types = self.scenario.object_metadata.object_types
    traj_data = np.array(
        self.traj.stack_fields(['x', 'y', 'length', 'width', 'yaw', 'z', 'height'])
    )  # Forces to np from jnp

    traj_data = traj_data[get_ego_mask(self.scenario) == 0]
    obj_types = obj_types[get_ego_mask(self.scenario) == 0]

    origin_x, origin_y, origin_yaw = get_ego_position(self.scenario, self.use_log_traj)
    num_obj, num_steps, _ = traj_data.shape

    traj_data[:,:,0] -= origin_x
    traj_data[:,:,1] -= origin_y
    traj_data[:,:,4] -= origin_yaw

    agent_dict = AgentDict()
    for obj_idx in range(num_obj):
      current_obj_data = traj_data[obj_idx, :, :] # [TS, 5]
      current_obj_data[:,:2] = rotate_nx2_array(current_obj_data[:,:2], origin_yaw)

      if obj_types[obj_idx] == 1:
        obj_type = 'Vehicle'
      elif obj_types[obj_idx] == 2:
        obj_type = 'Pedestrian'
      else:
        continue

      bboxes = {}
      for step in range(num_steps):
        box = current_obj_data[step, :]
        yaw = current_obj_data[step, 4]
        bboxes[step] = (get_3d_bbox(box), yaw) 

      agent_dict.insert_agent(agent_id=obj_idx, 
                              semantic=obj_type,
                              bboxes=bboxes)
    return agent_dict

  def get_ego_poses(self, scenario, use_log_traj=True, start=0, steps=-1):
    traj = scenario.log_trajectory if use_log_traj else scenario.sim_trajectory

    traj_xy = np.array(traj.xy[get_ego_mask(scenario) == 1, :, :].squeeze(0))
    traj_yaw = np.array(traj.yaw[get_ego_mask(scenario) == 1, :].squeeze(0))

    origin_x, origin_y, origin_yaw = get_ego_position(scenario, use_log_traj)
    traj_xy[:,0] -= origin_x
    traj_xy[:,1] -= origin_y
    traj_xy[:,:2] = rotate_nx2_array(traj_xy[:,:2], origin_yaw)
    traj_yaw -= origin_yaw

    poses = []
    num_timesteps = traj_xy.shape[0]
    for timestep in range(num_timesteps):
      x, y = traj_xy[timestep, :]
      yaw = traj_yaw[timestep]

      poses.append(x_y_yaw_to_pose(x, y, yaw))
    return convert_poses_to_relative(poses, relative='first')
  
  def create_map_sequence(self):
    maps = []
    state = self.scenario

    for _ in tqdm(range(state.remaining_timesteps+1), desc="Processing maps"):
      lidardm_map = get_lidardm_map(state, self.use_log_traj)
      map_encoded = encode(lidardm_map * 255)
      maps.append(map_encoded)
      state = state.replace(timestep=state.timestep+1)

    return maps
  
  def generate_mesh(self):
    maps = []
    poses = []
    for i, (encoded_map, pose) in enumerate(zip(self.maps, self.ego_poses)):
      if i == 0 or np.linalg.norm(poses[-1][:3,3] - pose[:3, 3]) > 35:

        decoded_map = decode(encoded_map, 13)
        decoded_map = decoded_map[:,:,:9]

        decoded_map = np.transpose(decoded_map, axes=(2, 0, 1))
        decoded_map = np.rot90(decoded_map, k=2, axes=(1,2))
        decoded_map = np.rot90(decoded_map, k=1, axes=(1, 2))

        maps.append(decoded_map.copy())
        poses.append(pose)
        # break

    mesh = None
    for idx in tqdm(range(len(maps)), desc="Generating scene"):
      decoded_map, pose = maps[idx], poses[idx]
      current_mesh = sample_from_map(torch.from_numpy(decoded_map).cuda().float(), self.model.cuda())

      # rotation = np.array([[0, 1,  0,  0],
      #                       [1, 0,  0,  0],
      #                       [0, 0,  1,  0],
      #                       [0, 0,  0,  1.0]])

      
      # current_mesh.transform(rotation)
      

      # triangles = np.asarray(current_mesh.triangles)
      # arr = np.vstack([triangles, np.flip(triangles, 1)])
      # current_mesh = o3d.geometry.TriangleMesh(current_mesh.vertices, o3d.utility.Vector3iVector(arr.copy()))
      # current_mesh.compute_vertex_normals()

      current_mesh.transform(pose)
      
      if mesh is None:
        mesh = current_mesh
      else: 
        mesh += current_mesh

    return mesh

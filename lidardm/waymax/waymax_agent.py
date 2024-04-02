import jax
from jax import numpy as jnp
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Callable
import cv2

import dataclasses
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax.datatypes import operations
from waymax.agents.expert import *

from lidardm.visualization import visualize_lidardm_map
from lidardm.waymax.waymax_map import get_lidardm_map
from lidardm.waymax.waymax_utils import *
from lidardm.waymax.path_generation.state_lattice_planner import get_path

@dataclasses.dataclass
class Traj:
    xs: List[float]
    ys: List[float]
    yaws: List[float]

class WaymaxSimpleSimulator():
  def __init__(self, 
               scenario: datatypes.SimulatorState, 
               num_agents: int = 32) -> None:
    '''
    Functionalities:
    - This class can simulate the behavior of the ego-vehicle given a pre-defined path
    - This class can assist the process of generating that path

    Inputs:
    - scenario: a SimulatorState from Waymax
    '''
    
    self.valids = jnp.asarray(np.ones(num_agents).astype(bool)[:,np.newaxis])
    self.scenario = scenario
    self.num_agents = num_agents

    # makss
    self.ego_mask = get_ego_mask(scenario)
    self.obj_mask = np.asarray(scenario.object_metadata.object_types)
    self.vel_mask = np.asarray(scenario.log_trajectory.vel_x[:,0] > 0) \
                  | np.asarray(scenario.log_trajectory.vel_y[:,0] > 0)

    # pre-defined data from lidardm
    self.width = 640
    self.m_coverage = 96

    # pre-defined data from waymax
    self.num_frames = 80
    self.start_frame = 10

    self.env_config = _config.EnvironmentConfig(
        # Ensure that the sim agent can control all valid objects.
        controlled_object=_config.ObjectType.VALID,
        max_num_objects=self.num_agents,
    )
    self.dynamics_model = dynamics.StateDynamics()
    self.env = _env.MultiAgentEnvironment(
        dynamics_model=self.dynamics_model,
        config=self.env_config,
    )

  def create_controllable_agent(self, 
                                trajectory: jax.Array,
                                is_controlled_func: Callable[
                                    [datatypes.SimulatorState], jax.Array
                                ],
  ) -> actor_core.WaymaxActorCore:
  
    def select_action( 
      params: actor_core.Params,
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
      
      """Infers an action for the current timestep given logged state."""
      del params, actor_state, rng  # unused.

      idx = state.timestep - self.start_frame
      data = jnp.tile(trajectory[idx], [self.num_agents, 1])
      logged_action = datatypes.Action(data=data, valid=self.valids)
      return actor_core.WaymaxActorOutput(
          actor_state=None,
          action=logged_action,
          is_controlled=is_controlled_func(state),
      )

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name='expert',
    )
  
  def generate_path(self, target_x: float, target_y: float, target_yaw: float) -> Traj:
    '''
    Functionalities:
    - generate a curved path starting from [0,0,0] and ending at the target location
    '''
    sequence = get_path([target_x, target_y, target_yaw])
    traj = Traj(sequence[0], sequence[1], sequence[2])
    return traj
  
  def _sample_sequence_speed(self, traj: Traj, s_vel: float, e_vel: float) -> Traj:
    '''
    Functionalities:
    - Given the trajectory from generated_path, it sample from the paths with known starting 
      and ending velocity to emulate smooth acceleration

    Inputs:
    - s_vel: starting velocity 
    - e_vel: ending velocity
    '''
    new_traj = Traj([traj.xs[0]], [traj.ys[0]], [traj.yaws[0]])
    
    d_total = (len(traj.xs)-1) * 0.1
    d_travel = 0
    accel = (e_vel**2 - s_vel**2)/(d_total * 2)
    
    while not np.isclose(d_total, d_travel, atol=0.2):
      current_vel = s_vel + accel * 0.1 * len(new_traj.xs)
      d_travel = 0.5 * (s_vel + current_vel) * 0.1 * len(new_traj.xs)
      
      d_idx = int(d_travel * 10)
      if d_idx >= len(traj.xs):
        new_traj.xs.append(traj.xs[-1])
        new_traj.ys.append(traj.ys[-1])
        new_traj.yaws.append(traj.yaws[-1])
        break    
      
      new_traj.xs.append(traj.xs[d_idx])
      new_traj.ys.append(traj.ys[d_idx])
      new_traj.yaws.append(traj.yaws[d_idx])
    return new_traj

  def sample_sequence_speed(self, traj: Traj, s_vel: float, e_vel: float, stop_middle_sec: float = 0) -> Traj:
    '''
    Functionalities:
    - Wrapper around _sample_sequence_speed that also supports stop midway, useful when 
      trying to emulate agent stopping midway before turning

    Inputs:
    - s_vel and e_vel: same as _sample_sequence_speed
    - stop_middle_sec: how long to stop for
    '''
    if stop_middle_sec <= 0:
      return self._sample_sequence_speed(traj, s_vel, e_vel)
    
    middle_idx = (len(traj.xs) - 1) // 2
    
    first_seq =  Traj(traj.xs[:middle_idx], traj.ys[:middle_idx], traj.yaws[:middle_idx])
    second_seq = Traj(traj.xs[middle_idx:], traj.ys[middle_idx:], traj.yaws[middle_idx:])

    first_seq = self._sample_sequence_speed(first_seq, s_vel, 0)
    second_seq = self._sample_sequence_speed(second_seq, 0, e_vel)

    xs   = first_seq[0] + [first_seq[0][-1]]*int(stop_middle_sec*10) + second_seq[0]
    ys   = first_seq[1] + [first_seq[1][-1]]*int(stop_middle_sec*10) + second_seq[1]
    yaws = first_seq[2] + [first_seq[2][-1]]*int(stop_middle_sec*10) + second_seq[2]

    return Traj(xs, ys, yaws)

  def pad_sequence(self, traj: Traj, target_yaw: float, speed: float) -> Traj:
    '''
    Functionalities:
    - Waymax expects trajectory to have 80 timesteps, this function pads timestep 
      with uniform velocity waypoints

    Inputs:
    - target_yaw: the yaw angle to pad the trajectory, usually the same as the 
                  target_yaw from generate_path()
    - speed: the speed to uniformly generated waypoints from
    '''
    assert len(traj.xs) == len(traj.ys) and len(traj.ys) == len(traj.yaws)

    if len(traj.xs) < self.num_frames:
      remaining = self.num_frames - len(traj.xs)
      for _ in range(remaining):
        traj.xs.append(traj.xs[-1] + np.cos(target_yaw) * speed * 0.1)
        traj.ys.append(traj.ys[-1] + np.sin(target_yaw) * speed * 0.1)
        traj.yaws.append(target_yaw)
            
    return traj

  def transform_to_frame(self, traj: Traj, 
                         real_origin: Tuple[float, float, float]
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Functionalities:
    - Every functions so far, abeit in cartesian unit, operates in local frame. This function transform that to 
      the frame of the ego-vehicle (or to the real origin)

    Inputs:
    - real_origin: the x, y, yaw to transform the frames to 

    Outputs:
    - Tuple of 3 ndarrays (xs, ys, yaws)
    '''
    origin_x, origin_y, origin_yaw = real_origin
    xs, ys, yaws = traj.xs, traj.ys, traj.yaws

    xs, ys = list(rotate_nx2_array(np.vstack((xs, ys)).T, -origin_yaw).T)
    yaws = np.array(yaws) + float(origin_yaw)

    xs = np.array(xs) + float(origin_x) 
    ys = np.array(ys) + float(origin_y) 

    return xs, ys, yaws
    
  def visualize_chosen_traj(self, traj: Traj) -> np.ndarray:
    '''
    Functionalities:
    - visualize the chosen trajectory on the first frame of the simulation, useful
      when tweaking the trajectory

    Output:
    - an image of the trajectory overlaid on the first frame
    '''
    base_image = visualize_lidardm_map(get_lidardm_map(self.scenario.replace(timestep=self.start_frame)).copy())
    
    xs = ((np.array(traj.xs) * (self.width / self.m_coverage)) + (self.width / 2)).round().astype(np.int32) 
    ys = (-(np.array(traj.ys) * (self.width / self.m_coverage)) + (self.width / 2)).round().astype(np.int32)

    pts = np.vstack((xs, ys)).T
    pts = pts.reshape((-1, 1, 2))

    image = cv2.polylines(base_image, [pts], False, [1,0,0], 2)

    return image
  
  def simulate_chosen_traj(self, traj: Traj) -> datatypes.SimulatorState:
    '''
    Functionalities:
    - perform simulation on the given trajectories, all other agents will be IDM unless
      it's first frame is static

    Output:
    - a new SimulatorState. Access simulated behavior with use_log_traj=False
    '''
    scenario = self.scenario
    origin = get_ego_position(scenario.replace(timestep=self.start_frame))
    xs, ys, yaws = self.transform_to_frame(traj, origin)
    ego_traj = jnp.asarray(np.vstack((xs, ys, yaws, np.zeros_like(xs), np.zeros_like(xs))).T)

    ego_vehicle_actors = self.create_controllable_agent(
      trajectory=ego_traj,
      is_controlled_func=lambda state: (self.ego_mask == 1)
    )

    normal_vehicles_actors = agents.IDMRoutePolicy(
        additional_lookahead_distance=50,
        additional_lookahead_points=50,
        # max_lookahead=
        max_accel=1,
        max_deccel=4,
        desired_vel=10,
        is_controlled_func=lambda state: (self.ego_mask == 0) \
                                       & (self.obj_mask == 1) \
                                       & (self.vel_mask == 0) 
    )

    normal_ped_actors = agents.IDMRoutePolicy(
        additional_lookahead_distance=50,
        additional_lookahead_points=50,
        max_accel=0.5,
        max_deccel=1,
        desired_vel=1,
        is_controlled_func=lambda state: (self.ego_mask == 0) \
                                       & (self.obj_mask == 2) \
                                       & (self.vel_mask == 0) 
    )

    static_actors = agents.create_constant_speed_actor(
        speed=0.0,
        dynamics_model=self.dynamics_model,
        is_controlled_func=lambda state: (self.ego_mask == 0) & (self.vel_mask == 1),
    )

    actors = [ego_vehicle_actors, normal_ped_actors, normal_vehicles_actors, static_actors]
    jit_step = jax.jit(self.env.step)
    jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

    states = [self.env.reset(scenario)]
    for _ in tqdm(range(states[0].remaining_timesteps), desc="Simulating"):
      current_state = states[-1]

      outputs = [
          jit_select_action({}, current_state, None, None)
          for jit_select_action in jit_select_action_list
      ]
      action = agents.merge_actions(outputs)
      next_state = jit_step(current_state, action)

      states.append(next_state)

    temp_scene = states[1]
    for i in range(1, temp_scene.remaining_timesteps):
      temp_scene = temp_scene.replace(
          sim_trajectory=operations.update_by_slice_in_dim(
              inputs=temp_scene.sim_trajectory,
              updates=states[i].sim_trajectory,
              inputs_start_idx=temp_scene.timestep + i,
              updates_start_idx=states[i].timestep,
              slice_size=1,
              axis=-1,
          ),
      )

    temp_scene = temp_scene.replace(
      sim_trajectory=operations.dynamic_slice(
            inputs=temp_scene.sim_trajectory, 
            start_index=temp_scene.timestep, 
            slice_size=temp_scene.remaining_timesteps, 
            axis=-1
        ),
      log_trajectory=operations.dynamic_slice(
            inputs=temp_scene.log_trajectory, 
            start_index=temp_scene.timestep, 
            slice_size=temp_scene.remaining_timesteps, 
            axis=-1
        ),
      log_traffic_light=operations.dynamic_slice(
            inputs=temp_scene.log_traffic_light, 
            start_index=temp_scene.timestep, 
            slice_size=temp_scene.remaining_timesteps, 
            axis=-1
        )
    )

    return temp_scene.replace(timestep=0)


  def visualize_sim_scenario(self, scenario):
    '''
    Functionalities:
    - visualize the simulation

    Output:
    - a sequence of image on each simulated frame
    '''
    maps = []
    state = scenario.replace(timestep=0)
    use_log_traj= False
    original_x, original_y, origin_yaw = get_ego_position(state, use_log_traj=use_log_traj)

    for _ in range(state.remaining_timesteps+1):
      lidardm_map = get_lidardm_map(state, use_log_traj, lock_first_frame=True)
      rgb = visualize_lidardm_map(lidardm_map, plot_origin=False)
      current_x, current_y, _ = get_ego_position(state, use_log_traj=use_log_traj)

      current_x -= original_x 
      current_y -= original_y
      origin = rotate_nx2_array(np.array([current_x, current_y]), origin_yaw)

      origin_x = (( (origin[0]) * (self.width / self.m_coverage)) + (self.width / 2)).round()
      origin_y = (( -(origin[1]) * (self.width / self.m_coverage)) + (self.width / 2)).round()
      
      rgb = cv2.circle(rgb, (int(origin_x), int(origin_y)), 5, (0,255,0), -1)

      maps.append(rgb)
      state = state.replace(timestep=state.timestep+1)

    return maps
    
    
    
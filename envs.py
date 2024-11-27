import numpy as np
import gymnasium as gym
from gymnasium import spaces

from highway_env.envs import HighwayEnv
from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


# paper's set maximum car speed
Vehicle.MAX_SPEED = 120 / 3.6 # 120 km/h in m/s


# overriding the environment reward function to adapt to the paper's rewards
# parameter "action" is also not used on the original definition
# original function: https://github.com/Farama-Foundation/HighwayEnv/blob/master/highway_env/envs/highway_env.py#L118
def _rewards(self, action: Action):
    rewards_flags = {
        "collision_reward": 0,
        "speed_interval_reward": 0,
        "headway_reward": 0,
        "on_road_reward": 0,
        "right_lane_reward": 0,
        "lane_change_reward": 0,

        "high_speed_reward": 0
    }

    # the paper indicates that "when two ICV come within a 10-meter distance of each other, the original speeds of the two vehicles will be replaced with their average speed"
    # there will only be 3 or 5 controlled vehicles, so, altough the following code has complexity O(n_vehicles**2), it's alright
    controlled_vehicles_x = [vehicle.to_dict()["x"] for vehicle in self.controlled_vehicles]
    for i in range(len(self.controlled_vehicles)):
        for j in range(i + 1, len(self.controlled_vehicles)):
            if (controlled_vehicles_x[i] - controlled_vehicles_x[j]) <= 10:
                avg_speed = (self.controlled_vehicles[i].speed + self.controlled_vehicles[j].speed) / 2
                self.controlled_vehicles[i].speed = avg_speed
                self.controlled_vehicles[j].speed = avg_speed

    for vehicle in self.controlled_vehicles:
        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        ) # 0 = left lane, 1 = center lane, 2 = right lane
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        # obtain the headway distance from the current agent to the closest agent in front of it (if the current agent is the furthest one, headway distance = 0)
        vehicles_in_front_x = list(filter(lambda x: x > vehicle.to_dict()["x"], controlled_vehicles_x))
        headway_distance = (vehicles_in_front_x[0] - vehicle.to_dict()["x"]) if len(vehicles_in_front_x) > 0 else 0

        rewards_flags["collision_reward"] += float(vehicle.crashed)
        rewards_flags["speed_interval_reward"] += float(self.config["reward_speed_range"][0] < vehicle.speed < self.config["reward_speed_range"][1])
        rewards_flags["headway_reward"] += float(self.config["reward_headway_range"][0] < headway_distance < self.config["reward_headway_range"][1])
        rewards_flags["on_road_reward"] += float(vehicle.on_road)
        rewards_flags["right_lane_reward"] += float(lane == 2)
        rewards_flags["lane_change_reward"] += float(vehicle.lane_offset[2] != 0)
        rewards_flags["high_speed_reward"] += np.clip(scaled_speed, 0, 1)
    
    # instead of on_road_reward being [0, len] it will stay in the [0,1] range, as the _reward function will multiply the final reward value by this constant
    rewards_flags["on_road_reward"] /= len(self.controlled_vehicles)
    return rewards_flags
# substitute the function
HighwayEnv._rewards = _rewards


# overriding the environment _is_terminated function since it returns True only considering the controlled_vehicles[0]
# new fuction definition only returns True if all controlled vehicles meet the terminal conditions
# original function: https://github.com/Farama-Foundation/HighwayEnv/blob/master/highway_env/envs/highway_env.py#L137
def _is_terminated(self) -> bool:
    return all([
        vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road)
        for vehicle in self.controlled_vehicles
    ])
HighwayEnv._is_terminated = _is_terminated


# environment settings
default_config = HighwayEnv.default_config()
default_config.update({
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
        }
    },

    "action": {
        "type": "MultiAgentAction",
        "action_config": {
        "type": "DiscreteMetaAction",
        }
    },

    "screen_width": 720,
    "screen_height": 240,

    # environment "physical" configuration
    "lanes_count": 3,
    "controlled_vehicles": 1,
    "vehicles_count": 30,
    "vehicles_density": 1,
    "initial_lane_id": None,
    "duration": 60,  # seconds
    "ego_spacing": 2,

    # paper reward values
    "collision_reward": -40,
    "speed_interval_reward": 10, "reward_speed_range": [103/3.6, 120/3.6],
    "headway_reward": 10, "reward_headway_range": [50, 70],
    "on_road_reward": 1,
    "right_lane_reward": -1,
    "lane_change_reward": -1,

    "normalize_reward": True,
    "offroad_terminal": False,
})

# custom definition of highway-env multi-agent envrionment that allows easier RL training with SB3
class MultiAgentHighwayEnv(gym.Env):
    def __init__(self, original_env, n_agents, image_obs, density, init_headway_distance):
        super(MultiAgentHighwayEnv, self).__init__()
        
        self.original_env: HighwayEnv = original_env
        self.n_agents = n_agents
        self.image_obs = image_obs
        self.density = density
        self.init_headway_distance = init_headway_distance
        
        # flatten action 
        n_actions = 1
        for action_space in self.original_env.action_space:
            n_actions *= action_space.n
        self.action_space = spaces.Discrete(n_actions)
        
        # stack 2D observation spaces
        obs_shape = self.original_env.observation_space[0].shape 
        stacked_shape = (n_agents, *obs_shape) # kinematics: (n_agents, V, F) / image_obs: (n_agents * stack_size, H, W)

        self.observation_space = spaces.Box(
            low=np.min(self.original_env.observation_space[0].low),
            high=np.max(self.original_env.observation_space[0].high),
            shape=stacked_shape,
            dtype=self.original_env.observation_space[0].dtype
        )

    def reset(self, **kwargs):
        obs, info = self.original_env.reset(**kwargs)

        if self.init_headway_distance != None:
            n_agents = self.n_agents
            for i, vehicle in enumerate(self.original_env.unwrapped.controlled_vehicles):
                vehicle.position[0] = np.float64(200 - (n_agents-1 - i) * self.init_headway_distance)

        return self._flatten_observation(obs), info

    def step(self, action):
        # convert the flat action back to tuple
        tuple_action = self._flat_to_tuple(action)
        obs, reward, done, truncated, info = self.original_env.step(tuple_action)
        return self._flatten_observation(obs), reward, done, truncated, info

    def _flatten_observation(self, obs):
        # stack the tuple of 2D arrays along the new first dimension
        return np.stack(obs, axis=0)

    def _flat_to_tuple(self, flat_action):
        # convert flat action back to tuple of actions
        actions = []
        for action_space in self.original_env.action_space:
            actions.append(flat_action % action_space.n)
            flat_action //= action_space.n
        return tuple(actions)
    
    def render(self):
        self.original_env.render()


def get_env(n_agents=1, image_obs=False, density=2, init_headway_distance=None):
    config = default_config.copy()
    config["controlled_vehicles"] = n_agents
    config["vehicles_density"] = density

    if image_obs:
        config["observation"]["observation_config"] = {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        }

    env = HighwayEnv(config=config, render_mode="rgb_array")
    
    if init_headway_distance != None:
        n_agents = len(env.unwrapped.controlled_vehicles)
        for i, vehicle in enumerate(env.unwrapped.controlled_vehicles):
            vehicle.position[0] = np.float64(200 - (n_agents-1 - i) * init_headway_distance)

    return env


def get_sb3_env(n_agents, image_obs, density, init_headway_distance):
    original_env = get_env(n_agents, image_obs, density, init_headway_distance)
    return MultiAgentHighwayEnv(original_env, n_agents, image_obs, density, init_headway_distance)

import gymnasium as gym
import highway_env 
import numpy as np


class SmoothRacerWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, jerk_penalty_weight: float = 0.15):
        super().__init__(env)
        self.jerk_penalty_weight = jerk_penalty_weight
        self.prev_steering = 0.0


    def reset(self, **kwargs):
        self.prev_steering = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_steering = float(action[0])

        steering_jerk = abs(current_steering - self.prev_steering)
        shaped_reward = reward - (self.jerk_penalty_weight * steering_jerk)

        self.prev_steering = current_steering
        info["steering_jerk"] = steering_jerk

        return obs, float(shaped_reward), terminated, truncated, info


def make_racing_env(render_mode: str = None) -> gym.Env:
    config = {
        "observation": {
            "type": "OccupancyGrid",
            "features": ["presence", "on_road"],
            "grid_size": [[-18, 18], [-18, 18]],
            "grid_step": [3, 3],
            "as_image": False,
            "align_to_vehicle_axes": True,
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": False,  # Cruise control handles throttle; agent learns steering
            "lateral": True,
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": 120,
        "collision_reward": -10.0,
        "lane_centering_cost": 4.0,
        "action_reward": -0.3,
        "controlled_vehicles": 1,
        "other_vehicles": 2,  # Set to 0 for pure time-trial, or 2+ for overtaking
    }
    env = gym.make("racetrack-v0", render_mode=render_mode, config=config)
    return SmoothRacerWrapper(env)
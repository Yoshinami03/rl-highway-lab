from pettingzoo.utils import ParallelEnv
import gymnasium as gym
import highway_env
import numpy as np
from gymnasium import spaces


class HighwayMultiEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "highway_multi_merge", "is_parallel": True}

    def __init__(self, num_agents=5, render_mode="human"):
        self.render_mode = render_mode
        self.env = gym.make(
            "highway-v0",
            render_mode=render_mode,
            config={
                "vehicles_count": num_agents,
                "controlled_vehicles": num_agents,
                "observation": {"type": "Kinematics"},
                "duration": 40,
            },
        )
        self.core_env = self.env.unwrapped
        self.possible_agents = [f"car_{i}" for i in range(num_agents)]
        self.num_cars = num_agents
        obs_low = np.array([-np.inf, -np.inf, 0.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        obs_space = spaces.Box(low=obs_low, high=obs_high, shape=(3,), dtype=np.float32)
        act_space = spaces.Discrete(5)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}
        self.agents = []

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        obs_dict = {a: self._get_vehicle_observation(i) for i, a in enumerate(self.agents)}
        info_dict = {a: {} for a in self.agents}
        return obs_dict, info_dict

    def step(self, actions):
        for i, a in enumerate(self.agents):
            action = actions.get(a, 1)
            v = self.core_env.road.vehicles[i]
            if action == 0:
                v.target_speed = max(0.0, v.target_speed - 1.0)
            elif action == 2:
                v.target_speed += 1.0
            elif action == 3:
                road, start, lane = v.target_lane_index
                v.target_lane_index = (road, start, max(0, lane - 1))
            elif action == 4:
                road, start, lane = v.target_lane_index
                v.target_lane_index = (road, start, lane + 1)

        _, _, term, trunc, _ = self.env.step(1)

        obs_dict = {a: self._get_vehicle_observation(i) for i, a in enumerate(self.agents)}
        rewards = {a: self._calc_reward(self.core_env.road.vehicles[i]) for i, a in enumerate(self.agents)}
        terminations = {a: bool(term) for a in self.agents}
        truncations = {a: bool(trunc) for a in self.agents}
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs_dict, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _get_vehicle_observation(self, i):
        v = self.core_env.road.vehicles[i]
        return np.array([v.position[0], v.position[1], v.speed], dtype=np.float32)

    def _calc_reward(self, v):
        r = v.speed / 30.0
        if getattr(v, "crashed", False):
            r -= 100.0
        return float(r)


if __name__ == "__main__":
    env = HighwayMultiEnv(num_agents=5)
    obs, info = env.reset()

    for _ in range(1000):
        actions = {agent: env.env.action_space.sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

        if all(terminations.values()) or all(truncations.values()):
            obs, info = env.reset()

    env.close()

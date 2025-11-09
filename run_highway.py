from pettingzoo import ParallelEnv
import gymnasium as gym
import highway_env
import numpy as np


class HighwayMultiEnv(ParallelEnv):
    def __init__(self, num_agents=5, render_mode="human"):
        self.env = gym.make("highway-v0", render_mode=render_mode, config={
            "vehicles_count": num_agents,
            "controlled_vehicles": num_agents,
            "observation": {"type": "Kinematics"},
            "duration": 40
        })
        self.agents = [f"car_{i}" for i in range(num_agents)]
        self.num_cars = num_agents

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_dict[agent] = self._get_vehicle_observation(i)
        info_dict = {agent: info for agent in self.agents}
        return obs_dict, info_dict

    def step(self, actions):
        """
        actions: dict[str, int]
          例: {"car_0": 0, "car_1": 2, ...}
        """
        obs_list, reward_list = {}, {}
        terminated, truncated = {}, {}
        info_dict = {}

        # 全エージェントを順番にstep
        for i, agent in enumerate(self.agents):
            action = actions.get(agent, self.env.action_space.sample())
            obs, reward, term, trunc, info = self.env.step(action)

            obs_list[agent] = obs
            reward_list[agent] = reward
            terminated[agent] = term
            truncated[agent] = trunc
            info_dict[agent] = info

        # 終了条件（全体）
        done = all(terminated.values()) or all(truncated.values())

        if done:
            obs, info = self.env.reset()
            obs_list = {agent: obs for agent in self.agents}
            info_dict = {agent: info for agent in self.agents}

        return obs_list, reward_list, terminated, truncated, info_dict

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        
    # 車両の観測
    def _get_vehicle_observation(self, index):
        v = self.env.road.vehicles[index]
        return np.array([v.position[0], v.position[1], v.spped], dtype=np.float32)

if __name__ == "__main__":
    env = HighwayMultiEnv(num_agents=5)
    obs, info = env.reset()

    for _ in range(1000):
        # 各エージェントがランダムに行動
        actions = {agent: env.env.action_space.sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

        if all(terminations.values()) or all(truncations.values()):
            obs, info = env.reset()

    env.close()

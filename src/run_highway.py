from typing import Dict, List, Optional, Tuple, Union
from pettingzoo.utils import ParallelEnv
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import highway_env
from env_config import HighwayEnvConfig, default_config


class HighwayMultiEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "highway_multi_merge", "is_parallel": True}

    def __init__(
        self, 
        config: Optional[HighwayEnvConfig] = None, 
        num_agents: Optional[int] = None, 
        render_mode: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            config: HighwayEnvConfigインスタンス（Noneの場合はdefault_configを使用）
            num_agents: エージェント数（configより優先、後方互換性のため）
            render_mode: レンダーモード（configより優先、後方互換性のため）
        """
        # 設定の適用（後方互換性のためnum_agentsとrender_modeも受け入れる）
        if config is None:
            config = default_config
        
        if num_agents is not None:
            config.num_agents = num_agents
        if render_mode is not None:
            config.render_mode = render_mode
        
        self.config = config
        self.render_mode = config.get_render_mode()
        
        # 環境の作成
        self.env = gym.make(
            config.env_name,
            render_mode=self.render_mode,
            config=config.get_gym_config(),
        )
        self.core_env = self.env.unwrapped
        self.possible_agents = [f"car_{i}" for i in range(config.num_agents)]
        
        # 観測空間の定義
        obs_low = np.array([-np.inf, -np.inf, 0.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        obs_space = spaces.Box(
            low=obs_low, 
            high=obs_high, 
            shape=config.obs_shape, 
            dtype=getattr(np, config.obs_dtype)
        )
        act_space = spaces.Discrete(config.action_space_size)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}
        self.agents = []

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    @property
    def unwrapped(self):
        return self

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        obs_dict = {a: self._get_vehicle_observation(i) for i, a in enumerate(self.agents)}
        info_dict = {a: {} for a in self.agents}
        return obs_dict, info_dict

    def step(
        self, 
        actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Dict]
    ]:
        for i, a in enumerate(self.agents):
            action = actions.get(a, 1)
            v = self.core_env.road.vehicles[i]
            if action == 0:
                v.target_speed = max(0.0, v.target_speed - 1.0)
            elif action == 2:
                v.target_speed += 1.0
            elif action == 3:
                road, start, lane = v.target_lane_index
                new_lane = max(0, lane - 1)
                v.target_lane_index = (road, start, new_lane)
            elif action == 4:
                road, start, lane = v.target_lane_index
                new_lane = lane + 1
                try:
                    # このレーンが存在しない場合は IndexError が出るので、そのときはレーン変更しない
                    self.core_env.road.network.get_lane((road, start, new_lane))
                    v.target_lane_index = (road, start, new_lane)
                except IndexError:
                    # 右端レーンなど、存在しないレーンに出ようとしたら無視
                    pass

        _, _, term, trunc, _ = self.env.step(1)

        obs_dict = {a: self._get_vehicle_observation(i) for i, a in enumerate(self.agents)}
        rewards = {a: self._calc_reward(self.core_env.road.vehicles[i]) for i, a in enumerate(self.agents)}
        terminations = {a: bool(term) for a in self.agents}
        truncations = {a: bool(trunc) for a in self.agents}
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs_dict, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _get_vehicle_observation(self, i: int) -> np.ndarray:
        if i >= len(self.core_env.road.vehicles):
            return np.zeros(self.config.obs_shape[0], dtype=getattr(np, self.config.obs_dtype))
            
        v = self.core_env.road.vehicles[i]
        return np.array([v.position[0], v.position[1], v.speed], dtype=getattr(np, self.config.obs_dtype))

    def _calc_reward(self, v) -> float:
        r = v.speed / self.config.speed_normalization
        if getattr(v, "crashed", False):
            r += self.config.crash_penalty
        return float(r)
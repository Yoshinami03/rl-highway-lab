from typing import Dict, List, Optional, Tuple, Union
from pettingzoo.utils import ParallelEnv
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from gymnasium import spaces
import highway_env
from env_config import HighwayEnvConfig, default_config

# カスタムmerge環境をインポート（src配下のカスタム環境）
import sys
import os
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# カスタム環境を登録（既存のmerge-v0を上書き）
# カスタム環境はsrc/highway_env/envs/merge_env.pyに存在
try:
    # カスタム環境を登録
    register(
        id="merge-v0-custom",
        entry_point="highway_env.envs.merge_env:MergeEnv",
    )
except Exception:
    # 既に登録されている場合は無視
    pass


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
        # カスタム環境を使用する場合は直接インスタンス化
        if config.env_name == "merge-v0-custom":
            from highway_env.envs.merge_env import MergeEnv
            self.env = MergeEnv(config=config.get_gym_config(), render_mode=self.render_mode)
        else:
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
        
        # 車両生成管理
        self.step_count = 0
        self.last_spawn_time = 0.0
        self.controlled_vehicle_indices = []  # 制御対象車両のインデックス

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
        
        # リセット時に制御対象車両を初期化
        # 最初のnum_agents台を制御対象とする
        self.controlled_vehicle_indices = list(range(min(self.config.num_agents, len(self.core_env.road.vehicles))))
        self.agents = [f"car_{i}" for i in self.controlled_vehicle_indices]
        self.step_count = 0
        self.last_spawn_time = 0.0
        
        obs_dict = {a: self._get_vehicle_observation(i) for i, a in zip(self.controlled_vehicle_indices, self.agents)}
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
        # 車両の継続生成をチェック
        self._spawn_vehicle_if_needed()
        
        # 各エージェントの行動を適用
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                agent_idx = self.agents.index(agent_id)
                vehicle_idx = self.controlled_vehicle_indices[agent_idx]
                if vehicle_idx < len(self.core_env.road.vehicles):
                    v = self.core_env.road.vehicles[vehicle_idx]
                    self._apply_action(v, action)

        _, _, term, trunc, _ = self.env.step(1)
        self.step_count += 1

        # 制御対象車両のインデックスを更新（車両が削除された場合に対応）
        self._update_controlled_vehicle_indices()

        obs_dict = {a: self._get_vehicle_observation(self.controlled_vehicle_indices[i]) 
                   for i, a in enumerate(self.agents) if i < len(self.controlled_vehicle_indices)}
        rewards = {a: self._calc_reward(self.core_env.road.vehicles[self.controlled_vehicle_indices[i]]) 
                  for i, a in enumerate(self.agents) if i < len(self.controlled_vehicle_indices)}
        terminations = {a: bool(term) for a in self.agents}
        truncations = {a: bool(trunc) for a in self.agents}
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
            self.controlled_vehicle_indices = []

        return obs_dict, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        # カメラ視点を合流地点に固定
        if hasattr(self.core_env, 'road') and hasattr(self.core_env.road, 'camera_position'):
            self.core_env.road.camera_position = np.array([
                self.config.camera_position_x,
                self.config.camera_position_y
            ])
        # highway-envのカメラ設定を直接変更
        if hasattr(self.core_env, 'road'):
            # カメラ位置を設定（highway-envの実装に依存）
            try:
                self.core_env.road.camera_offset = np.array([
                    self.config.camera_position_x,
                    self.config.camera_position_y
                ])
            except AttributeError:
                pass
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
    
    def _apply_action(self, vehicle, action: int) -> None:
        """車両に行動を適用"""
        if action == 0:
            vehicle.target_speed = max(0.0, vehicle.target_speed - 1.0)
        elif action == 2:
            vehicle.target_speed += 1.0
        elif action == 3:
            road, start, lane = vehicle.target_lane_index
            new_lane = max(0, lane - 1)
            vehicle.target_lane_index = (road, start, new_lane)
        elif action == 4:
            road, start, lane = vehicle.target_lane_index
            new_lane = lane + 1
            try:
                # このレーンが存在しない場合は IndexError が出るので、そのときはレーン変更しない
                self.core_env.road.network.get_lane((road, start, new_lane))
                vehicle.target_lane_index = (road, start, new_lane)
            except (IndexError, AttributeError):
                # 右端レーンなど、存在しないレーンに出ようとしたら無視
                pass
    
    def _spawn_vehicle_if_needed(self) -> None:
        """一定時間おきにスタート地点から車両を生成"""
        current_time = self.step_count * self.config.time_step
        
        if current_time - self.last_spawn_time >= self.config.vehicle_spawn_interval:
            # 新しい車両を生成
            try:
                # highway-envの車両生成ロジックを使用
                from highway_env.vehicle.controller import ControlledVehicle
                
                # スタート地点のレーンを取得（最初のレーン）
                if hasattr(self.core_env, 'road') and len(self.core_env.road.network.graph) > 0:
                    # 最初のレーンを取得
                    first_lane_key = None
                    for road_id in self.core_env.road.network.graph:
                        for start in self.core_env.road.network.graph[road_id]:
                            for lane_id in self.core_env.road.network.graph[road_id][start]:
                                first_lane_key = (road_id, start, lane_id)
                                break
                            if first_lane_key:
                                break
                        if first_lane_key:
                            break
                    
                    if first_lane_key:
                        lane = self.core_env.road.network.get_lane(first_lane_key)
                        if lane:
                            position = lane.position(self.config.vehicle_spawn_position_x, 0.0)
                            new_vehicle = ControlledVehicle(
                                self.core_env.road,
                                position,
                                speed=30.0
                            )
                            self.core_env.road.vehicles.append(new_vehicle)
                            
                            # 制御対象車両リストに追加（最大num_agents台まで）
                            if len(self.controlled_vehicle_indices) < self.config.num_agents:
                                new_index = len(self.core_env.road.vehicles) - 1
                                self.controlled_vehicle_indices.append(new_index)
                                new_agent_id = f"car_{new_index}"
                                self.agents.append(new_agent_id)
                            
                            self.last_spawn_time = current_time
            except Exception as e:
                # 車両生成に失敗した場合は無視（環境の実装に依存するため）
                pass
    
    def _update_controlled_vehicle_indices(self) -> None:
        """制御対象車両のインデックスを更新（削除された車両を除外）"""
        # 有効な車両インデックスのみを保持
        valid_indices = []
        valid_agents = []
        
        for i, vehicle_idx in enumerate(self.controlled_vehicle_indices):
            if vehicle_idx < len(self.core_env.road.vehicles):
                vehicle = self.core_env.road.vehicles[vehicle_idx]
                # 衝突や削除されていない車両のみ保持
                if not getattr(vehicle, "crashed", False):
                    valid_indices.append(vehicle_idx)
                    if i < len(self.agents):
                        valid_agents.append(self.agents[i])
        
        self.controlled_vehicle_indices = valid_indices
        self.agents = valid_agents
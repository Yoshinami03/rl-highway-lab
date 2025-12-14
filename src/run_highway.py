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
        
        # controlled vehiclesを取得（reset後に確定するが、ここで参照方法を準備）
        self._controlled_vehicles = None
        
        # エージェント数を固定（設定から取得、変更不可）
        # controlled_vehicles数が設定されていればそれを使用、なければnum_agentsを使用
        num_controlled = config.controlled_vehicles if config.controlled_vehicles is not None else config.num_agents
        self._num_agents = num_controlled  # プライベート属性として保存（ParallelEnvのnum_agentsプロパティと競合しないように）
        self.possible_agents = [f"car_{i}" for i in range(self._num_agents)]
        
        # 観測空間の定義（拡張版：速度、レーン、前方車距離、相対速度）
        # デフォルトは4次元だが、設定で上書き可能
        obs_dim = config.obs_shape[0] if len(config.obs_shape) > 0 else 4
        obs_low = np.array([-np.inf] * obs_dim, dtype=np.float32)
        obs_high = np.array([np.inf] * obs_dim, dtype=np.float32)
        # 前方車距離と相対速度は正の値になることが多いので、適切な範囲を設定
        if obs_dim >= 4:
            obs_low[2] = 0.0  # 前方車距離は0以上
            obs_high[2] = 200.0  # 最大距離
            obs_low[3] = -50.0  # 相対速度の範囲
            obs_high[3] = 50.0
        
        obs_space = spaces.Box(
            low=obs_low, 
            high=obs_high, 
            shape=(obs_dim,), 
            dtype=getattr(np, config.obs_dtype)
        )
        act_space = spaces.Discrete(config.action_space_size)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}
        self.agents = []
        
        # 観測次元を保存（一貫性のため）
        self.obs_dim = obs_dim

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    @property
    def unwrapped(self):
        return self

    def _get_controlled_vehicles(self) -> List:
        """controlled vehiclesを取得（版差を吸収）
        
        エージェント数（self._num_agents）に合わせて返す
        """
        if self._controlled_vehicles is not None:
            return self._controlled_vehicles
        
        # 様々な属性名を試す
        for attr_name in ["controlled_vehicles", "controlledVehicles"]:
            if hasattr(self.core_env, attr_name):
                vehicles = getattr(self.core_env, attr_name)
                if isinstance(vehicles, (list, tuple)) and len(vehicles) > 0:
                    vehicles_list = list(vehicles)
                    # エージェント数に合わせて調整
                    if len(vehicles_list) >= self._num_agents:
                        self._controlled_vehicles = vehicles_list[:self._num_agents]
                    else:
                        # 足りない場合は既存のリストを返す（警告は出さない、観測で対応）
                        self._controlled_vehicles = vehicles_list
                    return self._controlled_vehicles
        
        # 単一エージェントの場合、vehicle属性があることが多い
        if hasattr(self.core_env, "vehicle") and self.core_env.vehicle is not None:
            self._controlled_vehicles = [self.core_env.vehicle]
            return self._controlled_vehicles
        
        # フォールバック：road.vehiclesの先頭から取得
        if len(self.core_env.road.vehicles) > 0:
            # エージェント数に合わせる
            vehicles_list = self.core_env.road.vehicles[:self._num_agents]
            self._controlled_vehicles = vehicles_list
            return self._controlled_vehicles
        
        return []

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        
        # controlled vehiclesを再取得（reset後に更新される）
        self._controlled_vehicles = None
        controlled = self._get_controlled_vehicles()
        
        # エージェント数は固定（変更しない）
        # controlled vehicles数が足りない場合は警告を出すが、エージェント数は維持
        if controlled and len(controlled) < self._num_agents:
            # 警告：controlled vehicles数がエージェント数より少ない
            # この場合、一部のエージェントは制御対象外の車両を参照することになる
            pass
        
        # エージェント数を固定（possible_agentsは変更しない）
        self.agents = self.possible_agents[:]
        
        # 観測を生成（常に同じエージェント数、同じshape）
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
        """
        各エージェントが独立に行動を決定し、自分の車両を制御する
        
        重要: 各エージェントは独立に行動を決定し、自分の車両のみを制御します。
        これにより、同じモデルを複製して使う形になり、協調行動が自然に生まれます。
        """
        # controlled vehiclesを取得
        controlled = self._get_controlled_vehicles()
        if not controlled:
            controlled = self.core_env.road.vehicles[:len(self.agents)]
        
        # 各エージェントの行動をcontrolled vehiclesに適用
        # 各エージェントは独立に行動を決定し、自分の車両（controlled[i]）のみを制御
        # これにより、シングルエージェントが全車両を制御するのではなく、
        # 各エージェントが自分の車両を制御する形になる
        for i, a in enumerate(self.agents):
            if i >= len(controlled):
                continue
            # エージェントaが決定した行動を取得
            action = actions.get(a, 1)
            # エージェントiが制御する車両を取得（各エージェントは自分の車両のみを制御）
            v = controlled[i]
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
                except (IndexError, AttributeError):
                    # 右端レーンなど、存在しないレーンに出ようとしたら無視
                    pass

        # env.step()に正しいactionを渡す
        # 単一エージェントの場合は最初のエージェントのactionを使用
        # 複数エージェントの場合は配列で渡す（highway-envの実装に依存）
        if len(controlled) == 1:
            # 単一エージェントの場合
            first_action = actions.get(self.agents[0], 1) if self.agents else 1
            _, _, term, trunc, base_info = self.env.step(first_action)
        else:
            # 複数エージェントの場合（joint action）
            joint_action = [actions.get(a, 1) for a in self.agents]
            # highway-envが配列を受け付けるかどうかは実装依存
            # まずは単一値で試し、エラーが出たら配列に変更
            try:
                _, _, term, trunc, base_info = self.env.step(joint_action)
            except (TypeError, ValueError):
                # 配列が受け付けられない場合は最初のactionのみ使用
                first_action = actions.get(self.agents[0], 1) if self.agents else 1
                _, _, term, trunc, base_info = self.env.step(first_action)

        # 各エージェントの観測を生成（各エージェントは自分の観測のみを受け取る）
        obs_dict = {a: self._get_vehicle_observation(i) for i, a in enumerate(self.agents)}
        
        # 各エージェントの報酬を計算（各エージェントは自分の報酬のみを受け取る）
        rewards = {a: self._calc_reward(controlled[i] if i < len(controlled) else self.core_env.road.vehicles[i]) 
                   for i, a in enumerate(self.agents)}
        
        # 終了条件を各エージェントに配布
        terminations = {a: bool(term) for a in self.agents}
        truncations = {a: bool(trunc) for a in self.agents}
        
        # base_infoを各エージェントに配布
        infos = {a: base_info.copy() if base_info else {} for a in self.agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs_dict, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _get_vehicle_observation(self, i: int) -> np.ndarray:
        """車両の観測を取得（拡張版：前方車との距離・相対速度を含む）
        
        各エージェントは自分の観測のみを受け取ります。
        これにより、各エージェントが独立に行動を決定できます。
        
        Args:
            i: エージェントのインデックス（0から始まる）
        
        Returns:
            エージェントiの観測（常に同じshapeを返す）
        """
        controlled = self._get_controlled_vehicles()
        if not controlled:
            controlled = self.core_env.road.vehicles
        
        # 車両が存在するかチェック
        if i >= len(controlled) or i >= len(self.core_env.road.vehicles):
            # 車両が存在しない場合はデフォルト値で埋める
            # ただし、常に同じshapeを返す
            if self.obs_dim <= 3:
                return np.array([0.0, 0.0, 0.0], dtype=getattr(np, self.config.obs_dtype))
            else:
                # 4次元以上の場合：[速度=0, レーンID=0, 前方車距離=200, 相対速度=0]
                obs = np.array([0.0, 0.0, 200.0, 0.0], dtype=getattr(np, self.config.obs_dtype))
                if self.obs_dim > 4:
                    obs = np.pad(obs, (0, self.obs_dim - 4), mode='constant', constant_values=0.0)
                elif self.obs_dim < 4:
                    obs = obs[:self.obs_dim]
                return obs
        
        # 車両を取得（controlled vehicles優先、なければroad.vehiclesから）
        if i < len(controlled):
            v = controlled[i]
        else:
            v = self.core_env.road.vehicles[i]
        
        # 基本観測：速度、レーンID
        speed = float(v.speed)
        lane_id = float(v.lane_index[2] if hasattr(v, "lane_index") and v.lane_index else 0)
        
        if self.obs_dim <= 3:
            # 元の3次元観測（後方互換性）
            return np.array([v.position[0], v.position[1], speed], dtype=getattr(np, self.config.obs_dtype))
        
        # 拡張観測：前方車との距離と相対速度を計算
        headway = 200.0  # デフォルト値（前方車なし）
        rel_speed = 0.0
        
        # 同レーンまたは隣接レーンの前方車を探す
        v_pos = np.array(v.position)
        v_lane = v.lane_index[2] if hasattr(v, "lane_index") and v.lane_index else 0
        
        for other in self.core_env.road.vehicles:
            if other is v:
                continue
            
            other_pos = np.array(other.position)
            other_lane = other.lane_index[2] if hasattr(other, "lane_index") and other.lane_index else 0
            
            # 前方（x方向が大きい）かつ同レーンまたは隣接レーン
            dx = other_pos[0] - v_pos[0]
            dy = other_pos[1] - v_pos[1]
            
            if dx > 0 and abs(dy) < 5.0:  # 前方でy方向が近い
                distance = np.linalg.norm([dx, dy])
                if distance < headway:
                    headway = distance
                    rel_speed = float(other.speed - v.speed)
        
        # 観測ベクトル：[速度, レーンID, 前方車距離, 相対速度]
        obs = np.array([speed, lane_id, headway, rel_speed], dtype=getattr(np, self.config.obs_dtype))
        
        # 設定された次元数に合わせて調整（常に同じshapeを返す）
        if self.obs_dim > 4:
            # さらに拡張する場合は0で埋める
            obs = np.pad(obs, (0, self.obs_dim - 4), mode='constant', constant_values=0.0)
        elif self.obs_dim < 4:
            # 次元が小さい場合は切り詰める
            obs = obs[:self.obs_dim]
        
        return obs

    def _calc_reward(self, v) -> float:
        """報酬を計算（速度報酬 + 衝突ペナルティ + 危険状態ペナルティ）"""
        r = v.speed / self.config.speed_normalization
        
        # 衝突ペナルティ
        if getattr(v, "crashed", False):
            r += self.config.crash_penalty
        
        # 危険状態ペナルティ（前方車との距離が近すぎる場合）
        v_pos = np.array(v.position)
        min_headway = 200.0
        
        for other in self.core_env.road.vehicles:
            if other is v:
                continue
            
            other_pos = np.array(other.position)
            dx = other_pos[0] - v_pos[0]
            dy = other_pos[1] - v_pos[1]
            
            if dx > 0 and abs(dy) < 5.0:  # 前方でy方向が近い
                distance = np.linalg.norm([dx, dy])
                if distance < min_headway:
                    min_headway = distance
        
        # 車間距離が10m未満の場合はペナルティ
        if min_headway < 10.0:
            r -= 1.0
        
        # TTC（Time To Collision）が小さい場合のペナルティ
        if min_headway < 200.0:
            # 前方車がより遅い場合のみTTCを計算
            for other in self.core_env.road.vehicles:
                if other is v:
                    continue
                
                other_pos = np.array(other.position)
                dx = other_pos[0] - v_pos[0]
                dy = other_pos[1] - v_pos[1]
                
                if dx > 0 and abs(dy) < 5.0:
                    distance = np.linalg.norm([dx, dy])
                    rel_speed = v.speed - other.speed
                    if rel_speed > 0.1:  # 前方車の方が遅い
                        ttc = distance / rel_speed
                        if ttc < 2.0:  # 2秒以内に衝突する可能性
                            r -= 5.0
        
        return float(r)
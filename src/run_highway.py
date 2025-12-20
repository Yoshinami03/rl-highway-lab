import os
import sys
from typing import Dict, List, Optional, Tuple, Union
from pettingzoo.utils import ParallelEnv
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# src 配下のローカル highway_env を優先して import する
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_HE_PATH = os.path.join(_SRC_DIR, "highway_env")
for _p in (_LOCAL_HE_PATH, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import highway_env
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
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
        
        # 観測空間の定義（17次元：レーン情報、周囲車両情報）
        # デフォルトは17次元だが、設定で上書き可能
        obs_dim = config.obs_shape[0] if len(config.obs_shape) > 0 else 17
        obs_low = np.array([-np.inf] * obs_dim, dtype=np.float32)
        obs_high = np.array([np.inf] * obs_dim, dtype=np.float32)
        
        # 17次元観測の範囲設定
        if obs_dim >= 17:
            # 1-2: レーン移動可否 (0 or 1)
            obs_low[0:2] = 0.0
            obs_high[0:2] = 1.0
            # 3: 行き止まりフラグ (0 or 1)
            obs_low[2] = 0.0
            obs_high[2] = 1.0
            # 4: 行き止まりまでの距離
            obs_low[3] = 0.0
            obs_high[3] = 1000.0
            # 5: 自車速度
            obs_low[4] = 0.0
            obs_high[4] = 50.0
            # 6-17: 周囲車両の距離と速度
            for i in range(5, 17, 2):
                obs_low[i] = 0.0  # 距離
                obs_high[i] = 200.0
                obs_low[i+1] = 0.0  # 速度
                obs_high[i+1] = 50.0
        
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

        # 目標台数（NUM_AGENTS と vehicles_count の最大）
        target = max(self._num_agents, self.config.vehicles_count or 0)

        # 車両数を target 以上に補う（road.vehicles）
        self._ensure_vehicle_count(target)

        # controlled が target 未満なら追加生成
        controlled = self._get_controlled_vehicles()
        if len(controlled) < target:
            self._add_controlled_vehicles(target - len(controlled))
            controlled = self._get_controlled_vehicles()

        # controlled を NUM_AGENTS に合わせて切り詰め（超過時）
        if len(controlled) > self._num_agents:
            self._controlled_vehicles = controlled[:self._num_agents]
        
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
            if not hasattr(v, "target_lane_index"):
                continue  # target_lane_indexがない車両はスキップ
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
        
        # 全車両の報酬を計算して合計（各車両に同じ値を配る）
        total_reward = self._calc_total_reward()
        rewards = {a: total_reward for a in self.agents}
        
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
        """車両の観測を取得（17次元：レーン情報、周囲車両情報）
        
        17次元の観測:
        1. 左レーン移動可否 (0 or 1)
        2. 右レーン移動可否 (0 or 1)
        3. 行き止まりフラグ (0 or 1)
        4. 行き止まりまでの距離
        5. 自車速度
        6. 前方車距離
        7. 前方車速度
        8. 後方車距離
        9. 後方車速度
        10. 左前車距離
        11. 左前車速度
        12. 右前車距離
        13. 右前車速度
        14. 左後車距離
        15. 左後車速度
        16. 右後車距離
        17. 右後車速度
        
        Args:
            i: エージェントのインデックス（0から始まる）
        
        Returns:
            エージェントiの観測（常に同じshapeを返す）
        """
        controlled = self._get_controlled_vehicles()
        if not controlled:
            controlled = self.core_env.road.vehicles
        
        # 車両が存在しない場合のデフォルト観測
        default_obs = np.zeros(self.obs_dim, dtype=getattr(np, self.config.obs_dtype))
        if self.obs_dim >= 17:
            # デフォルト値を設定
            default_obs[3] = 1000.0  # 行き止まりまでの距離
            default_obs[5] = 200.0   # 前方車距離
            default_obs[7] = 200.0   # 後方車距離
            default_obs[9] = 200.0   # 左前車距離
            default_obs[11] = 200.0  # 右前車距離
            default_obs[13] = 200.0  # 左後車距離
            default_obs[15] = 200.0  # 右後車距離
        
        # 車両が存在するかチェック
        if i >= len(controlled) or i >= len(self.core_env.road.vehicles):
            return default_obs
        
        # 車両を取得
        if i < len(controlled):
            v = controlled[i]
        else:
            v = self.core_env.road.vehicles[i]
        
        # 17次元観測を構築
        if self.obs_dim >= 17:
            obs = np.zeros(17, dtype=getattr(np, self.config.obs_dtype))
            
            # 1-2: レーン移動可否
            obs[0] = 1.0 if self._check_lane_available(v, -1) else 0.0  # 左
            obs[1] = 1.0 if self._check_lane_available(v, 1) else 0.0   # 右
            
            # 3-4: 行き止まり情報
            is_dead_end, dead_end_dist = self._get_dead_end_info(v)
            obs[2] = 1.0 if is_dead_end else 0.0
            obs[3] = dead_end_dist
            
            # 5: 自車速度
            obs[4] = float(v.speed)
            
            # 6-7: 前方車
            front_dist, front_speed = self._get_nearest_vehicle_info(v, "front", 0)
            obs[5] = front_dist
            obs[6] = front_speed
            
            # 8-9: 後方車
            back_dist, back_speed = self._get_nearest_vehicle_info(v, "back", 0)
            obs[7] = back_dist
            obs[8] = back_speed
            
            # 10-11: 左前車
            left_front_dist, left_front_speed = self._get_nearest_vehicle_info(v, "front", -1)
            obs[9] = left_front_dist
            obs[10] = left_front_speed
            
            # 12-13: 右前車
            right_front_dist, right_front_speed = self._get_nearest_vehicle_info(v, "front", 1)
            obs[11] = right_front_dist
            obs[12] = right_front_speed
            
            # 14-15: 左後車
            left_back_dist, left_back_speed = self._get_nearest_vehicle_info(v, "back", -1)
            obs[13] = left_back_dist
            obs[14] = left_back_speed
            
            # 16-17: 右後車
            right_back_dist, right_back_speed = self._get_nearest_vehicle_info(v, "back", 1)
            obs[15] = right_back_dist
            obs[16] = right_back_speed
            
            # 設定された次元数に合わせて調整
            if self.obs_dim > 17:
                obs = np.pad(obs, (0, self.obs_dim - 17), mode='constant', constant_values=0.0)
            elif self.obs_dim < 17:
                obs = obs[:self.obs_dim]
            
            return obs
        else:
            # 17次元未満の場合は後方互換性のため簡易版を返す
            speed = float(v.speed)
            lane_id = float(v.lane_index[2] if hasattr(v, "lane_index") and v.lane_index else 0)
            
            if self.obs_dim <= 3:
                return np.array([v.position[0], v.position[1], speed], dtype=getattr(np, self.config.obs_dtype))
            else:
                # 4次元: [速度, レーンID, 前方車距離, 相対速度]
                front_dist, front_speed = self._get_nearest_vehicle_info(v, "front", 0)
                rel_speed = front_speed - speed if front_dist < 200.0 else 0.0
                obs = np.array([speed, lane_id, front_dist, rel_speed], dtype=getattr(np, self.config.obs_dtype))
                
                if self.obs_dim > 4:
                    obs = np.pad(obs, (0, self.obs_dim - 4), mode='constant', constant_values=0.0)
                elif self.obs_dim < 4:
                    obs = obs[:self.obs_dim]
                
                return obs

    def _check_lane_available(self, vehicle, lane_offset: int) -> bool:
        """レーン移動可否をチェック"""
        if not hasattr(vehicle, 'lane_index') or not vehicle.lane_index:
            return False
        
        try:
            road, start, lane = vehicle.lane_index
            target_lane = lane + lane_offset
            self.core_env.road.network.get_lane((road, start, target_lane))
            return True
        except (KeyError, IndexError, AttributeError):
            return False

    def _get_dead_end_info(self, vehicle) -> tuple:
        """行き止まり情報を取得"""
        if not hasattr(vehicle, 'lane_index') or not vehicle.lane_index:
            return (False, 1000.0)
        
        try:
            current_lane = self.core_env.road.network.get_lane(vehicle.lane_index)
            if hasattr(current_lane, 'forbidden') and current_lane.forbidden:
                v_pos = vehicle.position[0]
                dead_end_pos = 310.0
                distance = max(0.0, dead_end_pos - v_pos)
                return (True, distance)
        except (KeyError, IndexError, AttributeError):
            pass
        
        return (False, 1000.0)

    def _get_nearest_vehicle_info(self, vehicle, direction: str, lane_offset: int) -> tuple:
        """指定方向・レーンオフセットの最も近い車両の距離と速度を取得"""
        if not hasattr(vehicle, 'lane_index') or not vehicle.lane_index:
            return (200.0, 0.0)
        
        try:
            road, start, lane = vehicle.lane_index
            target_lane = lane + lane_offset
            self.core_env.road.network.get_lane((road, start, target_lane))
        except (KeyError, IndexError, AttributeError):
            return (200.0, 0.0)
        
        v_pos = np.array(vehicle.position)
        min_distance = 200.0
        target_speed = 0.0
        
        for other in self.core_env.road.vehicles:
            if other is vehicle:
                continue
            
            if not hasattr(other, 'lane_index') or not other.lane_index:
                continue
            
            if other.lane_index != (road, start, target_lane):
                continue
            
            other_pos = np.array(other.position)
            dx = other_pos[0] - v_pos[0]
            
            if direction == "front" and dx <= 0:
                continue
            if direction == "back" and dx >= 0:
                continue
            
            distance = abs(dx)
            if distance < min_distance:
                min_distance = distance
                target_speed = float(other.speed)
        
        return (min_distance, target_speed)

    def _calc_total_reward(self) -> float:
        """全車両の報酬を計算して合計（全車両で共有）
        
        報酬項目:
        - 衝突ペナルティ（全車両チェック）
        - ゴール報酬（x > 370）
        - 合流レーンからのゴール追加報酬
        - 速度報酬（全車両の平均速度）
        
        Returns:
            全車両で共有する報酬の合計
        """
        total_reward = 0.0
        goal_position = 370.0
        
        num_vehicles = len(self.core_env.road.vehicles)
        if num_vehicles == 0:
            return 0.0
        
        total_speed = 0.0
        collision_count = 0
        goal_count = 0
        merge_lane_goal_count = 0
        
        for vehicle in self.core_env.road.vehicles:
            # 衝突ペナルティ
            if getattr(vehicle, 'crashed', False):
                collision_count += 1
            
            # ゴール判定
            if hasattr(vehicle, 'position') and vehicle.position[0] > goal_position:
                goal_count += 1
                # 合流レーンからのゴールは追加ボーナス
                if hasattr(vehicle, 'lane_index') and vehicle.lane_index:
                    road_id = vehicle.lane_index[0]
                    if road_id in ["j", "k"]:
                        merge_lane_goal_count += 1
            
            # 速度を累積
            if hasattr(vehicle, 'speed'):
                total_speed += vehicle.speed
        
        # 報酬計算
        # 衝突ペナルティ: -100 per collision
        total_reward += collision_count * (-100.0)
        
        # ゴール報酬: +50 per goal
        total_reward += goal_count * 50.0
        
        # 合流レーンからのゴール追加報酬: +30
        total_reward += merge_lane_goal_count * 30.0
        
        # 速度報酬: 平均速度に基づく（0-30 m/s を 0-10 にスケール）
        avg_speed = total_speed / num_vehicles
        speed_reward = (avg_speed / 30.0) * 10.0
        total_reward += speed_reward
        
        return float(total_reward)

    def _calc_reward(self, v) -> float:
        """個別車両の報酬を計算（後方互換性のため残す）"""
        # 新しい実装では _calc_total_reward() を使用
        # この関数は互換性のためのみ残す
        return self._calc_total_reward()

    def _ensure_vehicle_count(self, target: Optional[int] = None) -> None:
        """target 台数以上の車両が存在するように補充する"""
        if target is None:
            target = max(self._num_agents, self.config.vehicles_count or self._num_agents)

        # 参照用のベース車両（先頭がなければ作らない）
        if not self.core_env.road.vehicles:
            return
        base = self.core_env.road.vehicles[0]

        while len(self.core_env.road.vehicles) < target:
            # 既存車両の少し後方に複製して配置（簡易配置）
            offset = -5.0 * (len(self.core_env.road.vehicles) - 0)
            new_pos = np.array(base.position) + np.array([offset, 0.0])
            new_vehicle = ControlledVehicle(
                self.core_env.road,
                position=new_pos,
                heading=base.heading if hasattr(base, "heading") else 0.0,
                speed=base.speed if hasattr(base, "speed") else 0.0,
            )
            new_vehicle.target_lane_index = getattr(base, "target_lane_index", None)
            self.core_env.road.vehicles.append(new_vehicle)

    def _add_controlled_vehicles(self, count: int) -> None:
        """不足しているcontrolled vehiclesを追加生成する（ランダムレーンでスポーン）"""
        if count <= 0:
            return

        road = self.core_env.road
        net = road.network

        # 利用可能なレーンを取得（3要素タプルのみを対象）
        lane_keys = [k for k in net.graph.keys() if isinstance(k, tuple) and len(k) == 3]
        if not lane_keys:
            return

        # ベース速度（先頭車両があればそれを使う）
        base_speed = 20.0
        if road.vehicles:
            base_speed = getattr(road.vehicles[0], "speed", base_speed)

        if self._controlled_vehicles is None:
            self._controlled_vehicles = []

        import random
        for _ in range(count):
            lane_key = random.choice(lane_keys)
            lane = net.get_lane(lane_key)

            # レーン始点付近にばらつきを持たせてスポーン（0〜30m の範囲）
            x = random.uniform(0, 30.0)
            position = lane.position(x, 0)
            heading = lane.heading_at(x)

            v = ControlledVehicle(
                road,
                position=position,
                heading=heading,
                speed=base_speed,
            )
            # レーンは選択したlane_key
            v.target_lane_index = lane_key

            road.vehicles.append(v)
            self._controlled_vehicles.append(v)
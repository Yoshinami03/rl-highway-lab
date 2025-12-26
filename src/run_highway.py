from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

from pettingzoo.utils import ParallelEnv
import gymnasium as gym
import numpy as np
from gymnasium import spaces

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_HE_PATH = os.path.join(_SRC_DIR, "highway_env")
for _p in (_LOCAL_HE_PATH, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import highway_env

from env_config import HighwayEnvConfig, default_config
from components.fleet import ControlledFleet
from components.spawn import VehicleSpawner
from components.action import EnvStepper, VehicleActionApplier
from components.metrics import AgentVehicleSelector, ObservationBuilder, RewardCalculator, TeamRewardCalculator


class HighwayMultiEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "highway_multi_merge", "is_parallel": True}

    def __init__(
    self,
    config: Optional[HighwayEnvConfig] = None,
    num_agents: Optional[int] = None,
    render_mode: Optional[str] = None,
):
        if config is None:
            config = default_config

        if num_agents is not None:
            config.num_agents = num_agents
        if render_mode is not None:
            config.render_mode = render_mode

        self.config = config
        self.render_mode = config.get_render_mode()

        self.env = gym.make(
            config.env_name,
            render_mode=self.render_mode,
            config=config.get_gym_config(),
        )
        self.core_env = self.env.unwrapped

        max_agents = int(config.num_agents)
        self._num_agents = max_agents
        self.possible_agents = [f"car_{i}" for i in range(self._num_agents)]
        self.agents: List[str] = []

        self.obs_dim = int(config.obs_shape[0] if len(config.obs_shape) > 0 else 4)

        self.observation_spaces = {a: self._build_obs_space() for a in self.possible_agents}
        # MultiDiscrete([3, 3]): [speed_action, lane_action]
        # speed_action: 0=減速, 1=維持, 2=加速
        # lane_action:  0=左変更, 1=維持, 2=右変更
        self.action_spaces = {a: spaces.MultiDiscrete([3, 3]) for a in self.possible_agents}

        self._current_step = 0

        self._fleet = ControlledFleet(agent_ids=self.possible_agents)
        self._spawner = VehicleSpawner(
            core_env=self.core_env,
            spawn_probability=float(config.spawn_probability),
            spawn_cooldown_steps=int(config.spawn_cooldown_steps),
        )
        self._action_applier = VehicleActionApplier(core_env=self.core_env)
        self._selector = AgentVehicleSelector(core_env=self.core_env)
        self._obs_builder = ObservationBuilder(
            core_env=self.core_env,
            obs_dim=self.obs_dim,
            obs_dtype=str(config.obs_dtype),
            merge_start=float(config.merge_start),
            merge_end=float(config.merge_end),
            lane_width=float(config.lane_width),
            max_obs_dist=float(config.max_obs_dist),
            max_deadend_obs=float(config.max_deadend_obs),
        )
        self._reward_calc = RewardCalculator(
            core_env=self.core_env,
            speed_normalization=float(config.speed_normalization),
            crash_penalty=float(config.crash_penalty),
        )

        # チーム報酬計算器
        self._team_reward_calc = TeamRewardCalculator(
            core_env=self.core_env,
            reward_goal=float(config.reward_goal),
            crash_penalty=float(config.crash_penalty),
            close_dist_threshold=float(config.close_dist_threshold),
            close_penalty_base=float(config.close_penalty_base),
            close_penalty_slope=float(config.close_penalty_slope),
            accel_penalty_scale=float(config.accel_penalty_scale),
            lane_change_penalty=float(config.lane_change_penalty),
        )

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)

        self._current_step = 0
        self._spawner.reset()
        self._fleet.reset()

        alive = list(self.core_env.road.vehicles)
        self._fleet.assign_new(alive)

        # プール制: エージェントリストは常に固定
        self.agents = self.possible_agents[:]

        # 全エージェントに対して観測を生成（非アクティブはダミー観測）
        obs = {}
        info = {}
        for a in self.agents:
            is_active = self._fleet.is_active(a)
            vehicle = self._fleet.vehicle_of(a)
            obs[a] = self._obs_builder.build(vehicle, is_active=is_active)
            info[a] = {"is_active": is_active}

        return obs, info

    def step(self, actions):
        self._current_step += 1

        # クールダウンを更新
        self._fleet.update_cooldowns()

        alive = list(self.core_env.road.vehicles)
        self._fleet.drop_missing(alive)

        # スポーン可能なエージェントがあれば新規車両を生成
        spawnable = self._fleet.spawnable_agents()
        if spawnable:
            spawned = list(self._spawner.spawn_continuous(step=self._current_step))
            self._fleet.assign_new(spawned)

        self._debug_control_coverage(self._current_step)

        # プール制: エージェントリストは常に固定
        self.agents = self.possible_agents[:]

        # アクティブなエージェントのみアクションを適用
        # デフォルトアクション: [1, 1] = 速度維持、レーン維持
        default_action = np.array([1, 1])
        action_info = {}

        for a in self.agents:
            if not self._fleet.is_active(a):
                continue  # 非アクティブはスキップ
            v = self._fleet.vehicle_of(a)
            if v is None:
                continue
            action = actions.get(a, default_action)
            info = self._action_applier._apply_one(v, action)
            action_info[a] = info

        _, _, term, trunc, base_info = self.env.step(1)

        # ゴール/デッドエンド/衝突の検出
        events = self._detect_events()

        # イベント発生した車両をプールに戻す
        for agent_id, event in events.items():
            if event["goal"] or event["crash"] or event["deadend"] or event["collision"]:
                # 車両を道路から削除
                vehicle = self._fleet.vehicle_of(agent_id)
                if vehicle is not None and vehicle in self.core_env.road.vehicles:
                    self.core_env.road.vehicles.remove(vehicle)
                self._fleet.deactivate_to_pool(agent_id, reason="event")

        alive2 = list(self.core_env.road.vehicles)
        self._fleet.drop_missing(alive2)

        # 全エージェントに対して観測・報酬を生成
        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # チーム報酬の集計
        team_goal_count = sum(1 for e in events.values() if e["goal"])
        team_crash_count = sum(1 for e in events.values() if e["crash"] or e["deadend"] or e["collision"])
        team_deadend_count = sum(1 for e in events.values() if e["deadend"])
        team_collision_count = sum(1 for e in events.values() if e["collision"])

        # アクティブなエージェントと車両の取得
        active_agents = self._fleet.assigned_agents()
        vehicles = {a: self._fleet.vehicle_of(a) for a in self.agents}

        # チーム報酬の計算（全エージェントに同一の報酬）
        common_reward = self._team_reward_calc.calc_team_reward(
            events=events,
            action_infos=action_info,
            vehicles=vehicles,
            active_agents=active_agents,
        )

        for a in self.agents:
            is_active = self._fleet.is_active(a)
            vehicle = self._fleet.vehicle_of(a)
            event = events.get(a, {"goal": False, "crash": False, "deadend": False, "collision": False})

            obs[a] = self._obs_builder.build(vehicle, is_active=is_active)

            # チーム報酬: 全エージェントに同一の報酬
            rewards[a] = common_reward

            terminations[a] = bool(term)
            truncations[a] = bool(trunc)
            infos[a] = {
                "is_active": is_active,
                "event_goal": event["goal"],
                "event_crash": event["crash"],
                "event_deadend": event["deadend"],
                "event_collision": event["collision"],
                "team_goal_count": team_goal_count,
                "team_crash_count": team_crash_count,
                "team_deadend_count": team_deadend_count,
                "team_collision_count": team_collision_count,
                "team_reward": common_reward,
                "action_info": action_info.get(a, {}),
                **(dict(base_info) if base_info else {}),
            }

        # エピソード終了時もエージェントリストは維持
        # （PettingZooの仕様上、終了時はagents=[]にする必要あり）
        if term or trunc:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _detect_events(self) -> Dict[str, Dict[str, bool]]:
        """ゴール到達、デッドエンド、衝突を検出"""
        events = {}
        goal_x = float(self.config.goal_x)
        merge_end = float(self.config.merge_end)

        for a in self.possible_agents:
            if not self._fleet.is_active(a):
                events[a] = {"goal": False, "crash": False, "deadend": False, "collision": False}
                continue

            vehicle = self._fleet.vehicle_of(a)
            if vehicle is None:
                events[a] = {"goal": False, "crash": False, "deadend": False, "collision": False}
                continue

            pos = getattr(vehicle, "position", (0.0, 0.0))
            x = float(pos[0])
            crashed = bool(getattr(vehicle, "crashed", False))
            lane_index = getattr(vehicle, "lane_index", None)

            # ゴール到達
            is_goal = x >= goal_x

            # デッドエンド（合流レーンで merge_end を超過）
            is_ramp = (
                lane_index is not None
                and len(lane_index) >= 2
                and lane_index[0] == "j"
                and lane_index[1] == "k"
            )
            is_deadend = is_ramp and x >= merge_end

            # 衝突
            is_collision = crashed

            events[a] = {
                "goal": is_goal,
                "crash": crashed or is_deadend,
                "deadend": is_deadend,
                "collision": is_collision,
            }

        return events


    def render(self) -> Optional[np.ndarray]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _build_obs_space(self) -> spaces.Box:
        """17次元の観測空間を構築

        観測空間の構成:
        [0]  is_active        - 車両がアクティブか (0/1)
        [1]  has_left_lane    - 左レーンの有無 (0/1)
        [2]  has_right_lane   - 右レーンの有無 (0/1)
        [3]  deadend_dist     - 合流レーン終端までの距離特徴量 [0, max_deadend_obs]
        [4]  self_speed       - 自車速度 (km/h) [0, 200]
        [5-6]   同一レーン前方: (速度 [0,200], 距離特徴量 [0,max_obs_dist])
        [7-8]   同一レーン後方: (速度 [0,200], 距離特徴量 [0,max_obs_dist])
        [9-10]  左レーン前方:   (速度 [0,200], 距離特徴量 [0,max_obs_dist])
        [11-12] 左レーン後方:   (速度 [0,200], 距離特徴量 [0,max_obs_dist])
        [13-14] 右レーン前方:   (速度 [0,200], 距離特徴量 [0,max_obs_dist])
        [15-16] 右レーン後方:   (速度 [0,200], 距離特徴量 [0,max_obs_dist])
        """
        obs_dim = self.obs_dim
        max_obs_dist = float(self.config.max_obs_dist)
        max_deadend_obs = float(self.config.max_deadend_obs)

        if obs_dim == 17:
            # 17次元観測空間の範囲を定義
            obs_low = np.array([
                0.0,    # [0] is_active
                0.0,    # [1] has_left_lane
                0.0,    # [2] has_right_lane
                0.0,    # [3] deadend_dist
                0.0,    # [4] self_speed
                0.0, 0.0,    # [5-6] same front
                0.0, 0.0,    # [7-8] same back
                0.0, 0.0,    # [9-10] left front
                0.0, 0.0,    # [11-12] left back
                0.0, 0.0,    # [13-14] right front
                0.0, 0.0,    # [15-16] right back
            ], dtype=np.float32)

            obs_high = np.array([
                1.0,    # [0] is_active
                1.0,    # [1] has_left_lane
                1.0,    # [2] has_right_lane
                max_deadend_obs,    # [3] deadend_dist
                200.0,  # [4] self_speed (km/h)
                200.0, max_obs_dist,    # [5-6] same front
                200.0, max_obs_dist,    # [7-8] same back
                200.0, max_obs_dist,    # [9-10] left front
                200.0, max_obs_dist,    # [11-12] left back
                200.0, max_obs_dist,    # [13-14] right front
                200.0, max_obs_dist,    # [15-16] right back
            ], dtype=np.float32)
        else:
            # 旧形式との互換性のためのフォールバック
            obs_low = np.array([-np.inf] * obs_dim, dtype=np.float32)
            obs_high = np.array([np.inf] * obs_dim, dtype=np.float32)

        return spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_dim,),
            dtype=getattr(np, self.config.obs_dtype),
        )

    def _obs_for_agent_index(self, index: int) -> np.ndarray:
        controlled = self._fleet.primary()
        vehicle = self._selector.select(index=index, controlled=controlled)
        return self._obs_builder.build(vehicle)

    def _reward_for_agent_index(self, index: int) -> float:
        controlled = self._fleet.primary()
        vehicle = self._selector.select(index=index, controlled=controlled)
        return self._reward_calc.calc(vehicle)

    def _debug_control_coverage(self, step: int) -> None:
        road_vehicles = list(self.core_env.road.vehicles)
        assigned_agents = self._fleet.assigned_agents()
        assigned_vehicles = [self._fleet.vehicle_of(a) for a in assigned_agents]
        assigned_vehicle_ids = {id(v) for v in assigned_vehicles if v is not None}

        unassigned = [v for v in road_vehicles if id(v) not in assigned_vehicle_ids]

        if unassigned:
            sample = unassigned[:5]
            print("[unassigned_sample]", [type(v).__name__ for v in sample], [id(v) for v in sample])

"""環境設定パラメータ"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CoopMergeConfig:
    """協調合流環境の設定パラメータ"""

    # geometry
    pre_merge: float = 400.0
    merge_length: float = 200.0
    post_merge: float = 400.0

    # simulation
    episode_horizon: int = 500
    dt: float = 1.0

    # lanes
    lane_width: float = 4.0
    main_lane_count: int = 3

    # speeds
    vmin_kmh: float = 80.0
    vmax_kmh: float = 150.0
    dv_kmh: float = 10.0
    vclip_min_kmh: float = 0.0

    # perception
    max_obs_dist: float = 200.0
    max_deadend_obs: float = 400.0
    adj_obs_unlock_x: float = 200.0

    # spawn
    spawn_min_dist: float = 2.0
    spawn_lane_cooldown_steps: int = 2
    spawn_lane_cooldown_variance: int = 2  # ランダムな分散（0～この値）
    spawn_attempts_per_step: int = 15
    max_spawns_per_step: int = 2  # 1ステップあたりの最大生成数
    agent_cooldown_min: int = 3  # エージェントクールダウンの最小値
    agent_cooldown_max: int = 8  # エージェントクールダウンの最大値

    # lane change
    lane_change_steps: int = 2

    # collision
    collision_radius: float = 2.5
    collision_dist: float = 2.0
    vehicle_length: Optional[float] = None

    # reward (team)
    reward_goal: float = 10.0
    reward_crash: float = -100.0

    close_dist_threshold: float = 20.0
    close_penalty_base: float = -40.0
    close_penalty_slope: float = 2.0

    accel_penalty_scale: float = 0.1
    lane_change_penalty: float = -0.1

    # reward (individual)
    individual_reward_goal: float = 5.0
    individual_reward_crash: float = -50.0
    individual_lane_change_penalty: float = -0.05

    @property
    def goal_x(self) -> float:
        return self.pre_merge + self.merge_length + self.post_merge

    @property
    def merge_start(self) -> float:
        return self.pre_merge

    @property
    def merge_end(self) -> float:
        return self.pre_merge + self.merge_length

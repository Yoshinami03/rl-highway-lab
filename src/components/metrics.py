from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 観測空間の定数
# =============================================================================
OBS_DIM = 17  # 観測空間の次元数

# 観測の最大距離
MAX_OBS_DIST = 200.0  # 前方/後方車両の最大観測距離 [m]
MAX_DEADEND_OBS = 400.0  # デッドエンド距離の最大観測距離 [m]

# 道路パラメータ（env_configから読み込む想定だが、デフォルト値を定義）
DEFAULT_MERGE_START = 400.0  # 合流開始x座標
DEFAULT_MERGE_END = 600.0  # 合流終了x座標（デッドエンド）
DEFAULT_LANE_WIDTH = 4.0  # レーン幅

# 合流レーンのインデックス（j→k レーン）
RAMP_LANE_INDEX = ("j", "k", 0)


@dataclass(frozen=True)
class AgentVehicleSelector:
    core_env: Any

    def select(self, index: int, controlled: Sequence[Any]) -> Optional[Any]:
        if index < len(controlled):
            return controlled[index]
        road = getattr(self.core_env, "road", None)
        vehicles = getattr(road, "vehicles", None) if road is not None else None
        if vehicles is None:
            return None
        if index < len(vehicles):
            return vehicles[index]
        return None


@dataclass
class ObservationBuilder:
    """
    17次元の観測空間を構築するクラス

    観測空間の構成:
    [0]  is_active        - 車両がアクティブか (0/1)
    [1]  has_left_lane    - 左レーンの有無 (0/1)
    [2]  has_right_lane   - 右レーンの有無 (0/1)
    [3]  deadend_dist     - 合流レーン終端までの距離特徴量
    [4]  self_speed       - 自車速度 (km/h)
    [5-6]   同一レーン前方: (速度, 距離特徴量)
    [7-8]   同一レーン後方: (速度, 距離特徴量)
    [9-10]  左レーン前方:   (速度, 距離特徴量)
    [11-12] 左レーン後方:   (速度, 距離特徴量)
    [13-14] 右レーン前方:   (速度, 距離特徴量)
    [15-16] 右レーン後方:   (速度, 距離特徴量)
    """
    core_env: Any
    obs_dim: int = OBS_DIM
    obs_dtype: str = "float32"

    # 道路パラメータ
    merge_start: float = DEFAULT_MERGE_START
    merge_end: float = DEFAULT_MERGE_END
    lane_width: float = DEFAULT_LANE_WIDTH
    max_obs_dist: float = MAX_OBS_DIST
    max_deadend_obs: float = MAX_DEADEND_OBS

    def build(self, vehicle: Optional[Any], is_active: bool = True) -> np.ndarray:
        """観測ベクトルを構築"""
        if vehicle is None or not is_active:
            return self._default()
        return self._from_vehicle(vehicle)

    def _default(self) -> np.ndarray:
        """非アクティブ時のダミー観測（is_active=0, 他は全て0）"""
        dtype = getattr(np, self.obs_dtype)
        return np.zeros((self.obs_dim,), dtype=dtype)

    def _from_vehicle(self, vehicle: Any) -> np.ndarray:
        """車両から17次元の観測を生成"""
        dtype = getattr(np, self.obs_dtype)

        # 基本情報の取得
        lane_index = getattr(vehicle, "lane_index", None)
        pos = getattr(vehicle, "position", (0.0, 0.0))
        x = float(pos[0])
        speed_mps = float(getattr(vehicle, "speed", 0.0))
        speed_kmh = speed_mps * 3.6  # m/s -> km/h

        # レーン情報
        is_ramp = self._is_ramp_lane(lane_index)
        lane_id = self._get_lane_id(lane_index)

        # [0] is_active
        is_active = 1.0

        # [1] has_left_lane, [2] has_right_lane
        has_left, has_right = self._get_adjacent_lanes(lane_id, x, is_ramp)

        # [3] deadend_dist (合流レーン終端までの距離特徴量)
        deadend_dist = self._get_deadend_dist_feature(x, is_ramp)

        # [4] self_speed (km/h)
        self_speed = speed_kmh

        # 隣接車両情報の取得
        same_lane_id = lane_id
        left_lane_id = lane_id - 1 if lane_id > 0 and not is_ramp else (2 if is_ramp else -1)
        right_lane_id = lane_id + 1 if not is_ramp and lane_id < 2 else (-1 if is_ramp else -1)

        # 合流レーンの場合、右レーンは存在しない
        if is_ramp:
            right_lane_id = -1
        # 本線レーン2の場合、右側に合流レーンがある（x < merge_end の場合のみ）
        elif lane_id == 2 and x < self.merge_end:
            right_lane_id = -2  # 特殊マーカー: 合流レーン

        # [5-8] 同一レーン前方/後方
        same_front_speed, same_front_dist = self._get_nearest_vehicle(vehicle, same_lane_id, True)
        same_back_speed, same_back_dist = self._get_nearest_vehicle(vehicle, same_lane_id, False)

        # [9-12] 左レーン前方/後方
        if left_lane_id >= 0:
            left_front_speed, left_front_dist = self._get_nearest_vehicle(vehicle, left_lane_id, True)
            left_back_speed, left_back_dist = self._get_nearest_vehicle(vehicle, left_lane_id, False)
        else:
            left_front_speed, left_front_dist = 0.0, 0.0
            left_back_speed, left_back_dist = 0.0, 0.0

        # [13-16] 右レーン前方/後方
        if right_lane_id >= 0:
            right_front_speed, right_front_dist = self._get_nearest_vehicle(vehicle, right_lane_id, True)
            right_back_speed, right_back_dist = self._get_nearest_vehicle(vehicle, right_lane_id, False)
        elif right_lane_id == -2:  # 合流レーン
            right_front_speed, right_front_dist = self._get_nearest_vehicle_ramp(vehicle, True)
            right_back_speed, right_back_dist = self._get_nearest_vehicle_ramp(vehicle, False)
        else:
            right_front_speed, right_front_dist = 0.0, 0.0
            right_back_speed, right_back_dist = 0.0, 0.0

        obs = np.array([
            is_active,           # [0]
            has_left,            # [1]
            has_right,           # [2]
            deadend_dist,        # [3]
            self_speed,          # [4]
            same_front_speed,    # [5]
            same_front_dist,     # [6]
            same_back_speed,     # [7]
            same_back_dist,      # [8]
            left_front_speed,    # [9]
            left_front_dist,     # [10]
            left_back_speed,     # [11]
            left_back_dist,      # [12]
            right_front_speed,   # [13]
            right_front_dist,    # [14]
            right_back_speed,    # [15]
            right_back_dist,     # [16]
        ], dtype=dtype)

        return obs

    def _is_ramp_lane(self, lane_index: Optional[Tuple]) -> bool:
        """合流レーンかどうかを判定"""
        if lane_index is None or len(lane_index) < 2:
            return False
        # 合流レーンは ("j", "k", 0) で始まる
        return lane_index[0] == "j" and lane_index[1] == "k"

    def _get_lane_id(self, lane_index: Optional[Tuple]) -> int:
        """レーンIDを取得（本線: 0, 1, 2、合流レーン: 3）"""
        if lane_index is None or len(lane_index) < 3:
            return 0
        if self._is_ramp_lane(lane_index):
            return 3  # 合流レーン
        return int(lane_index[2])

    def _get_adjacent_lanes(self, lane_id: int, x: float, is_ramp: bool) -> Tuple[float, float]:
        """隣接レーンの有無を取得"""
        if is_ramp:
            # 合流レーン: 左は本線レーン2（合流区間内のみ）、右はなし
            has_left = 1.0 if self.merge_start <= x < self.merge_end else 0.0
            has_right = 0.0
        else:
            # 本線レーン
            has_left = 1.0 if lane_id > 0 else 0.0
            # 右側は次のレーンがあるか、合流レーンがあるか
            if lane_id < 2:
                has_right = 1.0
            elif lane_id == 2 and x < self.merge_end:
                has_right = 1.0  # 合流レーンがある
            else:
                has_right = 0.0
        return has_left, has_right

    def _get_deadend_dist_feature(self, x: float, is_ramp: bool) -> float:
        """デッドエンド距離特徴量を計算

        特徴量 = max_deadend_obs - actual_dist (clipped)
        合流レーン以外は0を返す
        """
        if not is_ramp:
            return 0.0
        # 合流レーン終端までの距離
        dist = max(0.0, self.merge_end - x)
        dist = min(dist, self.max_deadend_obs)
        return self.max_deadend_obs - dist

    def _get_nearest_vehicle(
        self,
        vehicle: Any,
        target_lane_id: int,
        is_front: bool
    ) -> Tuple[float, float]:
        """指定レーンの最近接車両の情報を取得

        Returns:
            (速度 [km/h], 距離特徴量)
            距離特徴量 = max_obs_dist - actual_dist (clipped)
        """
        road = getattr(self.core_env, "road", None)
        vehicles = getattr(road, "vehicles", []) if road is not None else []

        v_pos = np.array(getattr(vehicle, "position", (0.0, 0.0)), dtype=float)

        best_dist = None
        best_speed = 0.0

        for other in vehicles:
            if other is vehicle:
                continue

            other_lane_index = getattr(other, "lane_index", None)
            other_lane_id = self._get_lane_id(other_lane_index)

            # 合流レーンは別処理
            if self._is_ramp_lane(other_lane_index):
                continue

            if other_lane_id != target_lane_id:
                continue

            other_pos = np.array(getattr(other, "position", (0.0, 0.0)), dtype=float)
            dx = float(other_pos[0] - v_pos[0])

            # 前方/後方のフィルタリング
            if is_front and dx <= 0:
                continue
            if not is_front and dx >= 0:
                continue

            dist = abs(dx)
            if dist > self.max_obs_dist:
                continue

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_speed = float(getattr(other, "speed", 0.0)) * 3.6  # km/h

        if best_dist is None:
            return 0.0, 0.0

        # 距離特徴量: max_obs_dist - actual_dist
        dist_feature = self.max_obs_dist - best_dist
        return best_speed, dist_feature

    def _get_nearest_vehicle_ramp(
        self,
        vehicle: Any,
        is_front: bool
    ) -> Tuple[float, float]:
        """合流レーンの最近接車両の情報を取得"""
        road = getattr(self.core_env, "road", None)
        vehicles = getattr(road, "vehicles", []) if road is not None else []

        v_pos = np.array(getattr(vehicle, "position", (0.0, 0.0)), dtype=float)

        best_dist = None
        best_speed = 0.0

        for other in vehicles:
            if other is vehicle:
                continue

            other_lane_index = getattr(other, "lane_index", None)
            if not self._is_ramp_lane(other_lane_index):
                continue

            other_pos = np.array(getattr(other, "position", (0.0, 0.0)), dtype=float)
            dx = float(other_pos[0] - v_pos[0])

            if is_front and dx <= 0:
                continue
            if not is_front and dx >= 0:
                continue

            dist = abs(dx)
            if dist > self.max_obs_dist:
                continue

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_speed = float(getattr(other, "speed", 0.0)) * 3.6

        if best_dist is None:
            return 0.0, 0.0

        dist_feature = self.max_obs_dist - best_dist
        return best_speed, dist_feature


@dataclass(frozen=True)
class RewardCalculator:
    """個別報酬計算器（旧実装、互換性のため維持）"""
    core_env: Any
    speed_normalization: float
    crash_penalty: float

    def calc(self, vehicle: Optional[Any]) -> float:
        if vehicle is None:
            return 0.0

        speed = float(getattr(vehicle, "speed", 0.0))
        r = speed / float(self.speed_normalization)

        if bool(getattr(vehicle, "crashed", False)):
            r += float(self.crash_penalty)

        headway, closing_speed = self._nearest_ahead(vehicle)
        if headway < 10.0:
            r -= 1.0

        ttc = self._ttc(headway, closing_speed)
        if ttc is not None and ttc < 2.0:
            r -= 5.0

        return float(r)

    def _nearest_ahead(self, vehicle: Any) -> Tuple[float, float]:
        road = getattr(self.core_env, "road", None)
        vehicles = getattr(road, "vehicles", []) if road is not None else []
        v_pos = np.array(getattr(vehicle, "position", (0.0, 0.0)), dtype=float)
        v_speed = float(getattr(vehicle, "speed", 0.0))

        best_dist = 200.0
        best_closing = 0.0

        for other in vehicles:
            if other is vehicle:
                continue
            other_pos = np.array(getattr(other, "position", (0.0, 0.0)), dtype=float)
            dx = float(other_pos[0] - v_pos[0])
            dy = float(other_pos[1] - v_pos[1])
            if dx <= 0.0 or abs(dy) >= 5.0:
                continue
            dist = float(np.linalg.norm([dx, dy]))
            if dist < best_dist:
                best_dist = dist
                best_closing = v_speed - float(getattr(other, "speed", 0.0))

        return best_dist, best_closing

    def _ttc(self, headway: float, closing_speed: float) -> Optional[float]:
        if headway >= 200.0:
            return None
        if closing_speed <= 0.1:
            return None
        return headway / closing_speed


@dataclass
class TeamRewardCalculator:
    """
    チーム報酬計算器（Highway2.ipynb準拠）

    報酬構成:
    1. ゴール報酬: +reward_goal per vehicle
    2. 衝突/デッドエンドペナルティ: +crash_penalty per vehicle
    3. 車間距離ペナルティ: 連続的（-40 + dist*2 when < 20m）
    4. 加減速ペナルティ: -accel_penalty_scale * |速度変化量|
    5. レーン変更ペナルティ: lane_change_penalty（合流以外）
    """
    core_env: Any

    # 報酬パラメータ
    reward_goal: float = 10.0
    crash_penalty: float = -100.0
    close_dist_threshold: float = 20.0
    close_penalty_base: float = -40.0
    close_penalty_slope: float = 2.0
    accel_penalty_scale: float = 0.1
    lane_change_penalty: float = -0.1

    def calc_team_reward(
        self,
        events: Dict[str, Dict[str, bool]],
        action_infos: Dict[str, Dict[str, Any]],
        vehicles: Dict[str, Optional[Any]],
        active_agents: Sequence[str],
    ) -> float:
        """
        チーム報酬を計算

        Args:
            events: 各エージェントのイベント情報
            action_infos: 各エージェントのアクション情報
            vehicles: 各エージェントの車両
            active_agents: アクティブなエージェントのリスト

        Returns:
            common_reward: 全エージェントに適用される共通報酬
        """
        common_reward = 0.0

        # 1. ゴール報酬
        goal_count = sum(1 for e in events.values() if e.get("goal", False))
        common_reward += self.reward_goal * goal_count

        # 2. 衝突/デッドエンドペナルティ
        crash_count = sum(
            1 for e in events.values()
            if e.get("crash", False) or e.get("deadend", False) or e.get("collision", False)
        )
        common_reward += self.crash_penalty * crash_count

        # 3. 車間距離ペナルティ（アクティブ車両のみ）
        common_reward += self._calc_close_front_penalty(vehicles, active_agents)

        # 4. 加減速ペナルティ
        total_abs_speed_delta = sum(
            abs(info.get("speed_delta", 0.0))
            for info in action_infos.values()
        )
        common_reward -= self.accel_penalty_scale * total_abs_speed_delta

        # 5. レーン変更ペナルティ（合流以外）
        non_merge_lane_changes = sum(
            1 for info in action_infos.values()
            if info.get("lane_change", 0) != 0 and not info.get("is_merge", False)
        )
        common_reward += self.lane_change_penalty * non_merge_lane_changes

        return float(common_reward)

    def _calc_close_front_penalty(
        self,
        vehicles: Dict[str, Optional[Any]],
        active_agents: Sequence[str],
    ) -> float:
        """車間距離ペナルティを計算"""
        penalty = 0.0

        for agent_id in active_agents:
            vehicle = vehicles.get(agent_id)
            if vehicle is None:
                continue

            front_dist = self._get_front_distance(vehicle)
            if front_dist is not None and front_dist < self.close_dist_threshold:
                # penalty = -40 + dist * 2
                penalty += self.close_penalty_base + self.close_penalty_slope * front_dist

        return penalty

    def _get_front_distance(self, vehicle: Any) -> Optional[float]:
        """前方車両までの距離を取得"""
        road = getattr(self.core_env, "road", None)
        all_vehicles = getattr(road, "vehicles", []) if road is not None else []

        v_pos = np.array(getattr(vehicle, "position", (0.0, 0.0)), dtype=float)
        lane_index = getattr(vehicle, "lane_index", None)

        best_dist = None

        for other in all_vehicles:
            if other is vehicle:
                continue

            # 同一レーンのみ対象
            other_lane = getattr(other, "lane_index", None)
            if lane_index is None or other_lane is None:
                continue
            if len(lane_index) < 3 or len(other_lane) < 3:
                continue
            if lane_index[2] != other_lane[2]:
                continue

            other_pos = np.array(getattr(other, "position", (0.0, 0.0)), dtype=float)
            dx = float(other_pos[0] - v_pos[0])

            # 前方のみ
            if dx <= 0:
                continue

            if best_dist is None or dx < best_dist:
                best_dist = dx

        return best_dist

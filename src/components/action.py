from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

LaneIndex = Tuple[str, str, int]

# 速度変化量 (km/h)
SPEED_DELTA_KMH = 10.0  # Highway2.ipynbに合わせて10km/h

# MultiDiscrete([3, 3]) のアクション定義
# speed_action: 0=減速, 1=維持, 2=加速
# lane_action:  0=左変更, 1=維持, 2=右変更


@dataclass
class VehicleActionApplier:
    """
    MultiDiscrete([3, 3]) 形式のアクションを車両に適用する

    アクション形式:
        [speed_action, lane_action]
        speed_action: 0=減速(-10km/h), 1=維持, 2=加速(+10km/h)
        lane_action:  0=左変更, 1=維持, 2=右変更
    """
    core_env: Any
    speed_delta_kmh: float = SPEED_DELTA_KMH

    def apply(
        self,
        actions: Dict[str, Union[np.ndarray, Tuple[int, int], int]],
        agents: Sequence[str],
        vehicles: Sequence[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        アクションを全エージェントに適用

        Returns:
            action_info: 各エージェントのアクション情報
                - speed_delta: 適用された速度変化量 [km/h]
                - lane_change: レーン変更の方向 (-1=左, 0=なし, 1=右)
                - is_merge: 合流レーンから本線への変更か
        """
        action_info = {}

        for i, agent in enumerate(agents):
            if i >= len(vehicles):
                break

            action = actions.get(agent)
            if action is None:
                action = np.array([1, 1])  # デフォルト: 維持

            info = self._apply_one(vehicles[i], action)
            action_info[agent] = info

        return action_info

    def _apply_one(
        self,
        vehicle: Any,
        action: Union[np.ndarray, Tuple[int, int], int]
    ) -> Dict[str, Any]:
        """
        単一車両にアクションを適用

        Returns:
            info: アクション適用結果
        """
        info = {
            "speed_delta": 0.0,
            "lane_change": 0,
            "is_merge": False,
        }

        if not hasattr(vehicle, "target_lane_index"):
            return info

        # アクションの解析
        if isinstance(action, (np.ndarray, tuple, list)):
            speed_action = int(action[0])
            lane_action = int(action[1])
        else:
            # 旧形式（Discrete(5)）との互換性
            speed_action, lane_action = self._convert_discrete_to_multi(int(action))

        # 速度アクションの適用
        speed_delta = self._apply_speed_action(vehicle, speed_action)
        info["speed_delta"] = speed_delta

        # レーンアクションの適用
        lane_change, is_merge = self._apply_lane_action(vehicle, lane_action)
        info["lane_change"] = lane_change
        info["is_merge"] = is_merge

        return info

    def _convert_discrete_to_multi(self, action: int) -> Tuple[int, int]:
        """旧Discrete(5)形式をMultiDiscrete([3,3])に変換"""
        # 0: 減速 -> (0, 1)
        # 1: 維持 -> (1, 1)
        # 2: 加速 -> (2, 1)
        # 3: 左   -> (1, 0)
        # 4: 右   -> (1, 2)
        mapping = {
            0: (0, 1),  # 減速
            1: (1, 1),  # 維持
            2: (2, 1),  # 加速
            3: (1, 0),  # 左
            4: (1, 2),  # 右
        }
        return mapping.get(action, (1, 1))

    def _apply_speed_action(self, vehicle: Any, speed_action: int) -> float:
        """速度アクションを適用"""
        current_speed = float(getattr(vehicle, "target_speed", 0.0))
        speed_mps = current_speed  # m/s

        # km/h -> m/s の変換
        delta_mps = self.speed_delta_kmh / 3.6

        if speed_action == 0:  # 減速
            new_speed = max(0.0, speed_mps - delta_mps)
            vehicle.target_speed = new_speed
            return -self.speed_delta_kmh
        elif speed_action == 2:  # 加速
            new_speed = speed_mps + delta_mps
            vehicle.target_speed = new_speed
            return self.speed_delta_kmh
        else:  # 維持
            return 0.0

    def _apply_lane_action(self, vehicle: Any, lane_action: int) -> Tuple[int, bool]:
        """
        レーンアクションを適用

        Returns:
            (lane_change, is_merge)
            lane_change: -1=左, 0=なし, 1=右
            is_merge: 合流レーンから本線への変更か
        """
        if lane_action == 1:  # 維持
            return 0, False

        road, start, lane = vehicle.target_lane_index
        is_ramp = (road == "j" and start == "k")

        if lane_action == 0:  # 左変更
            if is_ramp:
                # 合流レーンから本線レーン2へ（合流アクション）
                # 合流が可能かどうかはレーン存在確認で判断
                new_lane_index = ("b", "c", 1)  # 本線レーン2に相当
                if self._can_merge_to_main(vehicle):
                    vehicle.target_lane_index = new_lane_index
                    return -1, True
                return 0, False
            else:
                # 通常の左変更
                new_lane = max(0, int(lane) - 1)
                if new_lane != lane:
                    vehicle.target_lane_index = (road, start, new_lane)
                    return -1, False
                return 0, False

        elif lane_action == 2:  # 右変更
            if is_ramp:
                # 合流レーンでは右変更不可
                return 0, False
            else:
                new_lane = int(lane) + 1
                new_index = (road, start, new_lane)
                if self._lane_exists(new_index):
                    vehicle.target_lane_index = new_index
                    return 1, False
                return 0, False

        return 0, False

    def _can_merge_to_main(self, vehicle: Any) -> bool:
        """合流レーンから本線への合流が可能か確認"""
        # 合流区間内かどうかを確認
        pos = getattr(vehicle, "position", (0.0, 0.0))
        x = float(pos[0])

        # 合流区間の範囲（env_configから取得するのが理想だが、デフォルト値を使用）
        merge_start = 400.0
        merge_end = 600.0

        return merge_start <= x < merge_end

    def _lane_exists(self, lane_index: LaneIndex) -> bool:
        """レーンが存在するか確認"""
        road = getattr(self.core_env, "road", None)
        network = getattr(road, "network", None) if road is not None else None
        if network is None or not hasattr(network, "get_lane"):
            return False
        try:
            network.get_lane(lane_index)
            return True
        except (IndexError, AttributeError, KeyError):
            return False


@dataclass(frozen=True)
class EnvStepper:
    env: Any

    def step(self, actions: Dict[str, int], agents: Sequence[str], controlled_count: int):
        if controlled_count <= 1:
            first = agents[0] if agents else None
            a = int(actions.get(first, 1)) if first is not None else 1
            return self.env.step(a)

        joint = [int(actions.get(agent, 1)) for agent in agents]
        try:
            return self.env.step(joint)
        except (TypeError, ValueError):
            first = agents[0] if agents else None
            a = int(actions.get(first, 1)) if first is not None else 1
            return self.env.step(a)

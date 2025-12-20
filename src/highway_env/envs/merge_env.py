from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        """
        全車両の報酬を合計して返す（各車両に同じ値を配る）
        
        報酬項目:
        - 衝突ペナルティ（全車両分）
        - ゴール報酬（x > 370）
        - 合流レーンからのゴール追加報酬
        - 速度報酬（全車両の平均速度）

        :param action: the action performed
        :return: the total reward shared across all vehicles
        """
        total_reward = 0.0
        goal_position = 370.0
        
        # 全車両をチェック
        num_vehicles = len(self.road.vehicles)
        total_speed = 0.0
        collision_count = 0
        goal_count = 0
        merge_lane_goal_count = 0
        
        for vehicle in self.road.vehicles:
            # 衝突ペナルティ
            if getattr(vehicle, 'crashed', False):
                collision_count += 1
            
            # ゴール判定
            if hasattr(vehicle, 'position') and vehicle.position[0] > goal_position:
                goal_count += 1
                # 合流レーンからのゴールは追加ボーナス
                if hasattr(vehicle, 'lane_index') and vehicle.lane_index:
                    # 合流レーン（"j", "k"または"b", "c"のforbidden lane）からのゴール
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
        if num_vehicles > 0:
            avg_speed = total_speed / num_vehicles
            speed_reward = (avg_speed / 30.0) * 10.0
            total_reward += speed_reward
        
        return total_reward

    def _rewards(self, action: int) -> dict[str, float]:
        """
        報酬の詳細を辞書で返す（互換性のため残す）
        """
        total_reward = self._reward(action)
        return {
            "total_shared_reward": total_reward,
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _get_nearest_vehicle_info(self, vehicle, direction: str, lane_offset: int) -> tuple[float, float]:
        """
        指定方向・レーンオフセットの最も近い車両の距離と速度を取得
        
        Args:
            vehicle: 基準車両
            direction: "front" or "back"
            lane_offset: 0（同一レーン）, -1（左）, +1（右）
        
        Returns:
            (distance, speed): 車両なしなら (200.0, 0.0)
        """
        if not hasattr(vehicle, 'lane_index') or not vehicle.lane_index:
            return (200.0, 0.0)
        
        # 対象レーンを計算
        road, start, lane = vehicle.lane_index
        target_lane = lane + lane_offset
        
        # レーンが存在するかチェック
        try:
            target_lane_obj = self.road.network.get_lane((road, start, target_lane))
        except (KeyError, IndexError):
            return (200.0, 0.0)
        
        v_pos = np.array(vehicle.position)
        min_distance = 200.0
        target_speed = 0.0
        
        for other in self.road.vehicles:
            if other is vehicle:
                continue
            
            if not hasattr(other, 'lane_index') or not other.lane_index:
                continue
            
            # 同じレーンかチェック
            if other.lane_index != (road, start, target_lane):
                continue
            
            other_pos = np.array(other.position)
            dx = other_pos[0] - v_pos[0]
            
            # 方向チェック
            if direction == "front" and dx <= 0:
                continue
            if direction == "back" and dx >= 0:
                continue
            
            distance = abs(dx)
            if distance < min_distance:
                min_distance = distance
                target_speed = float(other.speed)
        
        return (min_distance, target_speed)

    def _check_lane_available(self, vehicle, lane_offset: int) -> bool:
        """
        指定レーンオフセットのレーンに移動可能かチェック
        
        Args:
            vehicle: 基準車両
            lane_offset: -1（左）, +1（右）
        
        Returns:
            移動可能ならTrue
        """
        if not hasattr(vehicle, 'lane_index') or not vehicle.lane_index:
            return False
        
        road, start, lane = vehicle.lane_index
        target_lane = lane + lane_offset
        
        # レーンが存在するかチェック
        try:
            target_lane_obj = self.road.network.get_lane((road, start, target_lane))
            return True
        except (KeyError, IndexError):
            return False

    def _get_dead_end_info(self, vehicle) -> tuple[bool, float]:
        """
        行き止まり情報を取得
        
        Args:
            vehicle: 基準車両
        
        Returns:
            (is_dead_end, distance): 行き止まりフラグと距離
        """
        if not hasattr(vehicle, 'lane_index') or not vehicle.lane_index:
            return (False, 1000.0)
        
        # 合流レーン（forbidden=True）にいる場合は行き止まり扱い
        try:
            current_lane = self.road.network.get_lane(vehicle.lane_index)
            if hasattr(current_lane, 'forbidden') and current_lane.forbidden:
                # 合流レーンの終端までの距離を計算
                v_pos = vehicle.position[0]
                # 合流レーンは約310m地点で終わる（ends[:3]の合計）
                dead_end_pos = 310.0
                distance = max(0.0, dead_end_pos - v_pos)
                return (True, distance)
        except (KeyError, IndexError):
            pass
        
        return (False, 1000.0)

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5.0, 5.0), 0.0)
            speed += self.np_random.uniform(-1.0, 1.0)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110.0, 0.0), speed=20.0
        )
        merging_v.target_speed = 30.0
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle

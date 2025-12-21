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
                "max_episode_steps": 500,
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["merging_speed_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )

    def _rewards(self, action: int) -> dict[str, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2)
                and isinstance(vehicle, ControlledVehicle)
            ),
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when max steps reached."""
        max_steps = self.config.get("max_episode_steps", 500)
        return self.vehicle.crashed or self.steps >= max_steps

    def _is_truncated(self) -> bool:
        return False

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
        Populate a road with several vehicles on the highway and on the merging lane.
        
        マルチエージェント対応：全ての車両をControlledVehicleとして生成
        - controlled_vehicles数に応じて制御対象車両を生成
        - 各車両は独立したエージェントによって制御される
        - 全エージェントが同じモデルを使用し、各自の観測で判断する

        :return: the ego-vehicle
        """
        road = self.road
        
        # 制御対象車両数を取得（デフォルト: 5）
        num_controlled = self.config.get("controlled_vehicles", 5)
        
        # 全ての車両をControlledVehicleとして生成
        self.controlled_vehicles = []
        
        # 本線レーン0に配置する車両数（半分）
        num_lane0 = num_controlled // 2
        # 本線レーン1に配置する車両数（残り半分）
        num_lane1 = num_controlled - num_lane0
        
        # 本線レーン0に車両を配置
        for i in range(num_lane0):
            position_x = 10.0 + i * 15.0  # 10m間隔で配置
            speed = self.np_random.uniform(25.0, 35.0)
            lane = road.network.get_lane(("a", "b", 0))
            position = lane.position(position_x, 0.0)
            
            vehicle = self.action_type.vehicle_class(
                road, position, speed=speed
            )
            vehicle.target_lane_index = ("a", "b", 0)
            road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
        
        # 本線レーン1に車両を配置
        for i in range(num_lane1):
            position_x = 10.0 + i * 15.0  # 10m間隔で配置
            speed = self.np_random.uniform(25.0, 35.0)
            lane = road.network.get_lane(("a", "b", 1))
            position = lane.position(position_x, 0.0)
            
            vehicle = self.action_type.vehicle_class(
                road, position, speed=speed
            )
            vehicle.target_lane_index = ("a", "b", 1)
            road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
        
        # 最初の車両をego-vehicleとして設定（後方互換性のため）
        self.vehicle = self.controlled_vehicles[0] if self.controlled_vehicles else None

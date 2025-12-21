from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

LaneIndex = Tuple[str, str, int]


@dataclass(frozen=True)
class VehicleActionApplier:
    core_env: Any

    def apply(self, actions: Dict[str, int], agents: Sequence[str], vehicles: Sequence[Any]) -> None:
        for i, agent in enumerate(agents):
            if i >= len(vehicles):
                return
            action = int(actions.get(agent, 1))
            self._apply_one(vehicles[i], action)

    def _apply_one(self, vehicle: Any, action: int) -> None:
        if not hasattr(vehicle, "target_lane_index"):
            return

        if action == 0:
            current = float(getattr(vehicle, "target_speed", 0.0))
            vehicle.target_speed = max(0.0, current - 1.0)
            return

        if action == 2:
            current = float(getattr(vehicle, "target_speed", 0.0))
            vehicle.target_speed = current + 1.0
            return

        if action == 3:
            road, start, lane = vehicle.target_lane_index
            vehicle.target_lane_index = (road, start, max(0, int(lane) - 1))
            return

        if action == 4:
            road, start, lane = vehicle.target_lane_index
            new_lane = int(lane) + 1
            if self._lane_exists((road, start, new_lane)):
                vehicle.target_lane_index = (road, start, new_lane)

    def _lane_exists(self, lane_index: LaneIndex) -> bool:
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

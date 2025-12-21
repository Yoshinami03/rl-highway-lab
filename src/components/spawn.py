from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from highway_env.vehicle.controller import ControlledVehicle

LaneIndex = Tuple[str, str, int]


@dataclass
class VehicleSpawner:
    core_env: Any
    spawn_probability: float
    spawn_cooldown_steps: int
    lanes: Sequence[LaneIndex] = (("j", "k", 0), ("a", "b", 0), ("a", "b", 1))
    lane_last_spawn_step: Dict[LaneIndex, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.lane_last_spawn_step = {lane: 0 for lane in self.lanes}

    def spawn_initial(self, step: int) -> Sequence[ControlledVehicle]:
        spawned = []
        v0 = self.spawn_at_start(("j", "k", 0), step)
        if v0 is not None:
            spawned.append(v0)

        rng = self._rng()
        if float(rng.random()) < 0.5:
            v1 = self.spawn_at_start(("a", "b", 0), step)
            if v1 is not None:
                spawned.append(v1)

        if float(rng.random()) < 0.5:
            v2 = self.spawn_at_start(("a", "b", 1), step)
            if v2 is not None:
                spawned.append(v2)

        return spawned

    def spawn_continuous(self, step: int) -> Sequence[ControlledVehicle]:
        rng = self._rng()
        spawned = []
        for lane in self.lanes:
            if float(rng.random()) >= float(self.spawn_probability):
                continue
            v = self.spawn_at_start(lane, step)
            if v is not None:
                spawned.append(v)
        return spawned

    def spawn_at_start(self, lane_index: LaneIndex, step: int) -> Optional[ControlledVehicle]:
        if not self._cooldown_ready(lane_index, step):
            return None
        if self._start_area_occupied(lane_index):
            return None

        lane = self.core_env.road.network.get_lane(lane_index)
        position = lane.position(0.0, 0.0)
        heading = lane.heading_at(0.0)

        rng = self._rng()
        speed = float(rng.uniform(25.0, 35.0))

        new_vehicle = ControlledVehicle(
            self.core_env.road,
            position=position,
            heading=heading,
            speed=speed,
        )
        new_vehicle.target_lane_index = lane_index
        new_vehicle.target_speed = 30.0

        self.core_env.road.vehicles.append(new_vehicle)
        self.lane_last_spawn_step[lane_index] = int(step)
        return new_vehicle

    def _cooldown_ready(self, lane_index: LaneIndex, step: int) -> bool:
        last = int(self.lane_last_spawn_step.get(lane_index, 0))
        return (int(step) - last) >= int(self.spawn_cooldown_steps)

    def _start_area_occupied(self, lane_index: LaneIndex) -> bool:
        for vehicle in self.core_env.road.vehicles:
            if getattr(vehicle, "lane_index", None) != lane_index:
                continue
            pos = getattr(vehicle, "position", None)
            if pos is None:
                continue
            x = float(pos[0])
            if 0.0 <= x <= 10.0:
                return True
        return False

    def _rng(self) -> Any:
        rng = getattr(self.core_env, "np_random", None)
        if rng is None:
            raise RuntimeError("core_env.np_random is missing")
        return rng

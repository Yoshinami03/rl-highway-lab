from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


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


@dataclass(frozen=True)
class ObservationBuilder:
    core_env: Any
    obs_dim: int
    obs_dtype: str

    def build(self, vehicle: Optional[Any]) -> np.ndarray:
        if vehicle is None:
            return self._default()
        return self._from_vehicle(vehicle)

    def _default(self) -> np.ndarray:
        dtype = getattr(np, self.obs_dtype)
        if self.obs_dim <= 3:
            return np.zeros((self.obs_dim,), dtype=dtype)
        base = np.array([0.0, 0.0, 200.0, 0.0], dtype=dtype)
        if self.obs_dim == 4:
            return base
        if self.obs_dim > 4:
            return np.pad(base, (0, self.obs_dim - 4), mode="constant", constant_values=0.0)
        return base[: self.obs_dim]

    def _from_vehicle(self, vehicle: Any) -> np.ndarray:
        dtype = getattr(np, self.obs_dtype)

        if self.obs_dim <= 3:
            pos = getattr(vehicle, "position", (0.0, 0.0))
            speed = float(getattr(vehicle, "speed", 0.0))
            base = np.array([float(pos[0]), float(pos[1]), speed], dtype=dtype)
            return base[: self.obs_dim]

        speed = float(getattr(vehicle, "speed", 0.0))
        lane_id = float(self._lane_id(vehicle))
        headway, rel_speed = self._headway_and_rel_speed(vehicle)
        base = np.array([speed, lane_id, headway, rel_speed], dtype=dtype)

        if self.obs_dim == 4:
            return base
        if self.obs_dim > 4:
            return np.pad(base, (0, self.obs_dim - 4), mode="constant", constant_values=0.0)
        return base[: self.obs_dim]

    def _lane_id(self, vehicle: Any) -> int:
        lane_index = getattr(vehicle, "lane_index", None)
        if not lane_index or len(lane_index) < 3:
            return 0
        return int(lane_index[2])

    def _headway_and_rel_speed(self, vehicle: Any) -> Tuple[float, float]:
        road = getattr(self.core_env, "road", None)
        vehicles = getattr(road, "vehicles", []) if road is not None else []
        v_pos = np.array(getattr(vehicle, "position", (0.0, 0.0)), dtype=float)
        v_speed = float(getattr(vehicle, "speed", 0.0))

        headway = 200.0
        rel_speed = 0.0

        for other in vehicles:
            if other is vehicle:
                continue
            other_pos = np.array(getattr(other, "position", (0.0, 0.0)), dtype=float)
            dx = float(other_pos[0] - v_pos[0])
            dy = float(other_pos[1] - v_pos[1])
            if dx <= 0.0 or abs(dy) >= 5.0:
                continue
            dist = float(np.linalg.norm([dx, dy]))
            if dist < headway:
                headway = dist
                rel_speed = float(getattr(other, "speed", 0.0)) - v_speed

        return headway, rel_speed


@dataclass(frozen=True)
class RewardCalculator:
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class ControlledFleet:
    max_agents: int
    vehicles: List[Any] = field(default_factory=list)

    def reset(self) -> None:
        self.vehicles = []

    def primary(self) -> List[Any]:
        if len(self.vehicles) >= self.max_agents:
            return self.vehicles[: self.max_agents]
        return self.vehicles

    def append(self, vehicle: Any) -> None:
        self.vehicles.append(vehicle)

    def is_empty(self) -> bool:
        return len(self.vehicles) == 0

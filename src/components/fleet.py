from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class ControlledFleet:
    agent_ids: Sequence[str]
    by_agent: Dict[str, Optional[Any]] = field(default_factory=dict)

    def reset(self) -> None:
        self.by_agent = {a: None for a in self.agent_ids}

    def assigned_agents(self) -> List[str]:
        return [a for a, v in self.by_agent.items() if v is not None]

    def free_agents(self) -> List[str]:
        return [a for a, v in self.by_agent.items() if v is None]

    def vehicle_of(self, agent_id: str) -> Optional[Any]:
        return self.by_agent.get(agent_id)

    def drop_missing(self, alive: Iterable[Any]) -> None:
        alive_set = set(alive)
        for a in self.agent_ids:
            v = self.by_agent.get(a)
            if v is None:
                continue
            if v not in alive_set:
                self.by_agent[a] = None

    def assign_new(self, vehicles: Sequence[Any]) -> None:
        free = self.free_agents()
        n = min(len(free), len(vehicles))
        for i in range(n):
            self.by_agent[free[i]] = vehicles[i]

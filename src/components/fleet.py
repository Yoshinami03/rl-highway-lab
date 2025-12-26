from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class ControlledFleet:
    """
    エージェント↔車両のマッピングを管理するプール制フリート

    Highway2.ipynbのようにエージェント数を固定し、
    非アクティブなエージェントにはダミー観測を返す設計。
    """
    agent_ids: Sequence[str]
    by_agent: Dict[str, Optional[Any]] = field(default_factory=dict)

    # プール制のための追加状態
    active: Dict[str, bool] = field(default_factory=dict)
    spawn_cooldown: Dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        """フリートをリセット"""
        self.by_agent = {a: None for a in self.agent_ids}
        self.active = {a: False for a in self.agent_ids}
        self.spawn_cooldown = {a: 0 for a in self.agent_ids}

    def is_active(self, agent_id: str) -> bool:
        """エージェントがアクティブかどうか"""
        return self.active.get(agent_id, False)

    def assigned_agents(self) -> List[str]:
        """車両がアサインされているエージェント（アクティブ）のリスト"""
        return [a for a, v in self.by_agent.items() if v is not None]

    def free_agents(self) -> List[str]:
        """車両がアサインされていないエージェント（非アクティブ）のリスト"""
        return [a for a, v in self.by_agent.items() if v is None]

    def spawnable_agents(self) -> List[str]:
        """スポーン可能なエージェント（非アクティブかつクールダウン完了）のリスト"""
        return [
            a for a in self.agent_ids
            if not self.active.get(a, False) and self.spawn_cooldown.get(a, 0) <= 0
        ]

    def vehicle_of(self, agent_id: str) -> Optional[Any]:
        """エージェントに割り当てられた車両を取得"""
        return self.by_agent.get(agent_id)

    def drop_missing(self, alive: Iterable[Any]) -> None:
        """生存していない車両を削除（プールに戻す）"""
        alive_set = set(alive)
        for a in self.agent_ids:
            v = self.by_agent.get(a)
            if v is None:
                continue
            if v not in alive_set:
                self._deactivate(a, reason="missing")

    def assign_new(self, vehicles: Sequence[Any]) -> List[str]:
        """新規車両をスポーン可能なエージェントに割り当て"""
        spawnable = self.spawnable_agents()
        assigned = []
        n = min(len(spawnable), len(vehicles))
        for i in range(n):
            agent_id = spawnable[i]
            self.by_agent[agent_id] = vehicles[i]
            self.active[agent_id] = True
            self.spawn_cooldown[agent_id] = 0
            assigned.append(agent_id)
        return assigned

    def deactivate_to_pool(self, agent_id: str, reason: str = "") -> None:
        """エージェントをプールに戻す（ゴール到達、衝突など）"""
        self._deactivate(agent_id, reason)

    def _deactivate(self, agent_id: str, reason: str = "") -> None:
        """エージェントを非アクティブ化"""
        self.by_agent[agent_id] = None
        self.active[agent_id] = False
        # 再スポーンまでのクールダウン（1ステップ）
        self.spawn_cooldown[agent_id] = 1

    def update_cooldowns(self) -> None:
        """クールダウンを更新（毎ステップ呼び出す）"""
        for a in self.agent_ids:
            if self.spawn_cooldown[a] > 0:
                self.spawn_cooldown[a] -= 1

    def primary(self) -> List[Any]:
        """アクティブな車両のリストを取得"""
        return [v for v in self.by_agent.values() if v is not None]

    def all_agents(self) -> List[str]:
        """全エージェントのリスト（プール制では常に固定）"""
        return list(self.agent_ids)

    def get_active_count(self) -> int:
        """アクティブなエージェント数"""
        return sum(1 for a in self.active.values() if a)

    def get_inactive_count(self) -> int:
        """非アクティブなエージェント数"""
        return sum(1 for a in self.active.values() if not a)

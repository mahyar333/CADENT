"""Deterministic finite automata that encode the task structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Hashable, Optional


@dataclass(frozen=True)
class DFA:
    """A DFA over event labels. Missing transitions leave the state unchanged."""

    transitions: Dict[int, Dict[Hashable, int]]
    start: int = 0
    accepting: FrozenSet[int] = field(default_factory=frozenset)

    def step(self, q: int, label: Optional[Hashable]) -> int:
        return self.transitions.get(q, {}).get(label, q)

    def is_accepting(self, q: int) -> bool:
        return q in self.accepting

    @property
    def states(self):
        return sorted(self.transitions)

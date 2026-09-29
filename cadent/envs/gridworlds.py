"""The Blind Craftsman grid world.

It follows the paper's experiments: four moves, an event is emitted whenever the agent
stands on a labelled cell after a move (also when a move is blocked), and the task progress is
tracked by a DFA. States are ``(position, inventory, automaton state)``.

With ``progress_rewards=False`` the reward is sparse: only task completion is rewarded, and every
other step costs the step penalty (the automaton-progress bonuses are removed).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from ..automaton import DFA
from .base import ProductEnv, State

Pos = Tuple[int, int]
MOVES = ("up", "down", "left", "right")


def move(pos: Pos, action: str, size: int, obstacles) -> Pos:
    x, y = pos
    nxt = {"up": (x - 1, y), "down": (x + 1, y), "left": (x, y - 1), "right": (x, y + 1)}[action]
    if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in obstacles:
        return nxt
    return pos


# --------------------------------------------------------------------------- Blind Craftsman
CRAFTSMAN_DFA = DFA(
    transitions={0: {"wood": 1}, 1: {"factory": 2, "wood": 3}, 2: {"wood": 4}, 3: {"factory": 5},
                 4: {"factory": 6, "wood": 7}, 5: {"wood": 8}, 6: {"wood": 9}, 7: {"factory": 10},
                 8: {"factory": 11}, 9: {"factory": 12}, 10: {"home": 13}, 11: {"home": 14},
                 12: {"home": 15}, 13: {}, 14: {}, 15: {}},
    start=0, accepting=frozenset({13, 14, 15}))


class BlindCraftsman(ProductEnv):
    """Collect wood (carry at most 2), craft tools at the factory (at most 3), then go home.

    Inventory is ``(wood, product)``; the factory converts all carried wood into products.
    """

    actions = MOVES
    dfa = CRAFTSMAN_DFA

    def __init__(self, size: int, labels: Dict[Pos, str], obstacles: Iterable[Pos] = (),
                 max_steps: int = 2500, name: str = "blind_craftsman", progress_rewards: bool = True):
        self.size, self.labels, self.obstacles = size, dict(labels), set(obstacles)
        self.max_steps, self.name, self.progress_rewards = max_steps, name, progress_rewards

    def reset(self) -> State:
        return ((0, 0), (0, 0), self.dfa.start)

    def outcomes(self, state: State, action: int) -> List[Tuple[State, float]]:
        pos, (wood, product), q = state
        nxt = move(pos, self.actions[action], self.size, self.obstacles)
        label = self.labels.get(nxt)
        q2 = self.dfa.step(q, label)
        if label == "wood" and wood < 2:
            wood += 1
        elif label == "factory" and wood > 0 and product < 3:
            craft = min(wood, 3 - product)
            product, wood = product + craft, wood - craft
        return [((nxt, (wood, product), q2), 1.0)]

    def reward(self, state: State, next_state: State) -> float:
        q, q2 = state[-1], next_state[-1]
        if self.dfa.is_accepting(q2):
            return 25.0
        return 1.0 if (q2 != q and self.progress_rewards) else -0.05

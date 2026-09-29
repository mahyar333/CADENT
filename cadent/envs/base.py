"""Common interface of the product environments (environment state x automaton state)."""

from __future__ import annotations

from typing import Hashable, List, Sequence, Tuple

import numpy as np

from ..automaton import DFA

State = Tuple[Hashable, ...]


class ProductEnv:
    """Base class. Subclasses implement ``reset``, ``outcomes`` and ``reward``.

    ``outcomes(state, action)`` lists every possible next state with its probability, which
    lets the teacher identify the state-action pairs that trigger an automaton transition
    (Eq. 4 of the paper). ``step`` samples one outcome.
    """

    name: str = "env"
    actions: Sequence[str] = ()
    dfa: DFA
    max_steps: int = 1000

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def reset(self) -> State:  # pragma: no cover - abstract
        raise NotImplementedError

    def outcomes(self, state: State, action: int) -> List[Tuple[State, float]]:  # pragma: no cover
        raise NotImplementedError

    def reward(self, state: State, next_state: State) -> float:  # pragma: no cover
        raise NotImplementedError

    def automaton_state(self, state: State) -> int:
        return state[-1]

    def is_terminal(self, state: State) -> bool:
        return self.dfa.is_accepting(self.automaton_state(state))

    def step(self, state: State, action: int, rng: np.random.Generator) -> Tuple[State, float]:
        outs = self.outcomes(state, action)
        if len(outs) == 1:
            nxt = outs[0][0]
        else:
            u, acc = rng.random(), 0.0
            nxt = outs[-1][0]
            for s, p in outs:
                acc += p
                if u < acc:
                    nxt = s
                    break
        return nxt, self.reward(state, nxt)

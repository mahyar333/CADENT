"""Teacher training on the source task and distillation of its knowledge.

The teacher is a tabular Q-learner on the source product MDP. Two kinds of knowledge are
extracted from its Q-values:

* strategic knowledge, the automaton transition values (Eq. 4)
  ``Q_AD(q, q') = mean { Q_T(s, a) : s has automaton state q and a leads to automaton state q' }``;
* tactical knowledge, the phase-conditioned policy (Eq. 3)
  ``pi_T(a | q) = softmax( mean_{s : q(s) = q} Q_T(s, .) / tau )``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .envs.base import ProductEnv


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (x - np.max(x)) / temperature
    e = np.exp(z)
    return e / e.sum()


def greedy(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with random tie-breaking."""
    best = np.flatnonzero(values == values.max())
    return int(best[0] if len(best) == 1 else rng.choice(best))


@dataclass
class TeacherConfig:
    episodes: int = 500
    alpha: float = 0.5
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99
    temperature: float = 1.0      # tau of Eq. 3


@dataclass
class TeacherKnowledge:
    q_ad: Dict[Tuple[int, int], float]      # automaton transition values, including self-transitions
    policy: Dict[int, np.ndarray]           # pi_T(. | q)
    success_rate: float                     # teacher success over its last 50 training episodes
    q_table: Dict = None                    # the teacher's Q-values (to re-distil at another tau)

    def with_temperature(self, env: ProductEnv, temperature: float) -> "TeacherKnowledge":
        """The same teacher with its policy distilled at temperature ``tau``."""
        _, policy = distill(env, self.q_table, temperature)
        return TeacherKnowledge(self.q_ad, policy, self.success_rate, self.q_table)


def train_teacher(env: ProductEnv, cfg: TeacherConfig, rng: np.random.Generator):
    """Q-learning on the source task. Returns the Q-table and the recent success rate."""
    q = defaultdict(lambda: np.zeros(env.n_actions))
    eps, successes = cfg.epsilon_start, []
    for _ in range(cfg.episodes):
        s = env.reset()
        for _ in range(env.max_steps):
            a = int(rng.integers(env.n_actions)) if rng.random() < eps else greedy(q[s], rng)
            s2, r = env.step(s, a, rng)
            target = r if env.is_terminal(s2) else r + cfg.gamma * q[s2].max()
            q[s][a] += cfg.alpha * (target - q[s][a])
            s = s2
            if env.is_terminal(s):
                break
        successes.append(env.is_terminal(s))
        eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
    return dict(q), float(np.mean(successes[-50:]))


def distill(env: ProductEnv, q_table: Dict, temperature: float = 1.0) -> Tuple[dict, dict]:
    """Eq. 4 (automaton transition values) and Eq. 3 (phase-conditioned teacher policy)."""
    per_transition, per_phase = defaultdict(list), defaultdict(list)
    for s, values in q_table.items():
        q = env.automaton_state(s)
        per_phase[q].append(values)
        if env.is_terminal(s):
            continue
        for a in range(env.n_actions):
            for q2 in {env.automaton_state(s2) for s2, p in env.outcomes(s, a) if p > 0}:
                per_transition[(q, q2)].append(values[a])
    q_ad = {k: float(np.mean(v)) for k, v in per_transition.items()}
    policy = {q: softmax(np.mean(v, axis=0), temperature) for q, v in per_phase.items()}
    return q_ad, policy


def teacher_knowledge(env: ProductEnv, cfg: TeacherConfig, rng: np.random.Generator) -> TeacherKnowledge:
    q_table, success = train_teacher(env, cfg, rng)
    q_ad, policy = distill(env, q_table, cfg.temperature)
    return TeacherKnowledge(q_ad, policy, success, q_table)

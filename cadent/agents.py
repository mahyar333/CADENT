"""Student agents: CADENT and the transfer baselines.

All students are tabular Q-learners on the target product MDP and share the same exploration
schedule; they differ only in how they use the teacher's knowledge.

* ``NoTransfer``: plain Q-learning.
* ``AutomatonDistillation`` (AD, Singireddy et al., 2023): the TD target of an automaton
  transition ``(q, q')`` is blended with the teacher's value ``Q_AD(q, q')`` using a weight
  ``rho ** n(q, q')`` that decays with the number of times the student made that transition.
* ``PolicyDistillation`` (PD, Rusu et al., 2016): a distillation term ``beta (pi_T - pi_S)`` pulls
  the student's softmax policy towards the teacher's policy.
* ``Kickstarting`` (Schmitt et al., 2018): the PD term with a single global weight annealed from 1.0
  to 0.1 over training.
* ``CADENT``: the experience-gated combination of the student's own update with the teacher's
  strategic (automaton-level) and tactical (policy-level) guidance. ``strategic="target"`` (default)
  also distils the automaton values into the student's TD target as AD does, so that at full trust
  CADENT reduces to AD and the gate adds PD-style guidance on top; ``strategic="reward"`` is the
  update exactly as written in the paper (Eqs. 5-11), where the automaton values enter only as the
  bonus ``r_AD`` inside the teacher term.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .envs.base import ProductEnv
from .teacher import TeacherKnowledge, greedy, softmax


@dataclass
class QLearningConfig:
    alpha: float = 0.6
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995      # per episode, shared by all methods


@dataclass
class EpisodeLog:
    rewards: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    success: List[int] = field(default_factory=list)
    trust: List[float] = field(default_factory=list)     # CADENT: mean omega over the episode


class Student:
    name = "No Transfer"

    def __init__(self, env: ProductEnv, cfg: QLearningConfig, rng: np.random.Generator):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.q = defaultdict(lambda: np.zeros(env.n_actions))
        self.epsilon = cfg.epsilon_start
        self.episode = 0
        self._trust: List[float] = []

    # ------------------------------------------------------------------ acting
    def act(self, s) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.n_actions))
        return greedy(self.q[s], self.rng)

    def td_error(self, s, a, r, s2, done) -> float:
        target = r if done else r + self.cfg.gamma * self.q[s2].max()
        return target - self.q[s][a]

    def learn(self, s, a, r, s2, done) -> None:
        self.q[s][a] += self.cfg.alpha * self.td_error(s, a, r, s2, done)

    def end_episode(self) -> None:
        self.episode += 1
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    # ------------------------------------------------------------------ training loop
    def train(self, episodes: int) -> EpisodeLog:
        env, log = self.env, EpisodeLog()
        for _ in range(episodes):
            s, total, steps = env.reset(), 0.0, 0
            self._trust = []
            for steps in range(1, env.max_steps + 1):
                a = self.act(s)
                s2, r = env.step(s, a, self.rng)
                done = env.is_terminal(s2)
                self.learn(s, a, r, s2, done)
                total += r
                s = s2
                if done:
                    break
            log.rewards.append(total)
            log.steps.append(steps)
            log.success.append(int(env.is_terminal(s)))
            log.trust.append(float(np.mean(self._trust)) if self._trust else float("nan"))
            self.end_episode()
        return log


class NoTransfer(Student):
    name = "No Transfer"


class AutomatonDistillation(Student):
    name = "Automaton Distillation"

    def __init__(self, env, cfg, rng, teacher: TeacherKnowledge, rho: float = 0.99):
        super().__init__(env, cfg, rng)
        self.q_ad, self.rho = teacher.q_ad, rho
        self.counts: Dict = defaultdict(int)

    def learn(self, s, a, r, s2, done) -> None:
        t = (self.env.automaton_state(s), self.env.automaton_state(s2))
        target = r if done else r + self.cfg.gamma * self.q[s2].max()
        if t in self.q_ad:
            self.counts[t] += 1
            w = self.rho ** self.counts[t]
            target = w * self.q_ad[t] + (1.0 - w) * target
        self.q[s][a] += self.cfg.alpha * (target - self.q[s][a])


class PolicyDistillation(Student):
    name = "Policy Distillation"

    def __init__(self, env, cfg, rng, teacher: TeacherKnowledge, beta: float = 0.4,
                 temperature: float = 1.0):
        super().__init__(env, cfg, rng)
        self.policy, self.beta, self.temperature = teacher.policy, beta, temperature

    def distill_weight(self) -> float:
        return self.beta

    def learn(self, s, a, r, s2, done) -> None:
        update = self.cfg.alpha * self.td_error(s, a, r, s2, done)
        pi_t = self.policy.get(self.env.automaton_state(s))
        if pi_t is not None:
            pi_s = softmax(self.q[s], self.temperature)
            update += self.distill_weight() * (pi_t[a] - pi_s[a])
        self.q[s][a] += update


class Kickstarting(PolicyDistillation):
    """PD with a global distillation weight annealed exponentially from 1.0 to 0.1.

    The weight reaches 0.1 after ``anneal_fraction`` of the training episodes and stays there.
    """

    name = "Kickstarting"

    def __init__(self, env, cfg, rng, teacher, beta: float = 0.4, temperature: float = 1.0,
                 anneal_fraction: float = 0.5, total_episodes: int = 1000,
                 initial_weight: float = 1.0, final_weight: float = 0.1):
        super().__init__(env, cfg, rng, teacher, beta, temperature)
        self.initial_weight, self.final_weight = initial_weight, final_weight
        self.anneal_episodes = max(1.0, anneal_fraction * total_episodes)

    def weight(self) -> float:
        frac = min(1.0, self.episode / self.anneal_episodes)
        return self.initial_weight * (self.final_weight / self.initial_weight) ** frac

    def distill_weight(self) -> float:
        return self.beta * self.weight()


class CADENT(Student):
    """Context-Aware Distillation with Experience-gated Transfer (Algorithm 1 of the paper).

    ``trust="gated"`` is the full method; ``trust="fixed"`` sets omega to ``fixed_trust`` for every
    state-action pair (ablation). ``lambda_ad=0`` or ``lambda_pd=0`` remove the strategic or the
    tactical guidance (ablations).
    """

    name = "CADENT"

    def __init__(self, env, cfg, rng, teacher: TeacherKnowledge, eta: float = 0.01, k: float = 5.0,
                 theta: float = 0.5, lambda_ad: float = 0.1, lambda_pd: float = 0.6,
                 trust: str = "gated", fixed_trust: float = 0.5, v_init: float = 0.0,
                 strategic: str = "target", rho: float = 0.9998, name: Optional[str] = None):
        super().__init__(env, cfg, rng)
        self.q_ad, self.policy = teacher.q_ad, teacher.policy
        self.eta, self.k, self.theta = eta, k, theta
        self.lambda_ad, self.lambda_pd = lambda_ad, lambda_pd
        self.trust_mode, self.fixed_trust = trust, fixed_trust
        self.strategic, self.rho = strategic, rho
        self.counts: Dict = defaultdict(int)
        # V(s, a). Algorithm 1 initialises it to 0; v_init > theta instead makes unvisited pairs start
        # with low trust, the early-training behaviour described in Section 4.2 of the paper.
        self.volatility: Dict = defaultdict(lambda: v_init)
        if name:
            self.name = name

    def trust(self, s, a, delta: float) -> float:
        if self.trust_mode == "fixed":
            return self.fixed_trust
        v = (1.0 - self.eta) * self.volatility[(s, a)] + self.eta * abs(delta)      # Eq. 8
        self.volatility[(s, a)] = v
        z = self.k * (v - self.theta)
        return float(1.0 / (1.0 + np.exp(min(z, 700.0))))                           # Eq. 9

    def student_error(self, s, a, r, s2, done, transition) -> float:
        """The student's own error: plain TD, or TD towards the AD-distilled target."""
        target = r if done else r + self.cfg.gamma * self.q[s2].max()
        if self.strategic == "target" and transition in self.q_ad:
            self.counts[transition] += 1
            w = self.rho ** self.counts[transition]
            target = w * self.q_ad[transition] + (1.0 - w) * target
        return target - self.q[s][a]

    def learn(self, s, a, r, s2, done) -> None:
        transition = (self.env.automaton_state(s), self.env.automaton_state(s2))
        delta = self.student_error(s, a, r, s2, done, transition)
        omega = self.trust(s, a, delta)
        self._trust.append(omega)
        q, q2 = self.env.automaton_state(s), self.env.automaton_state(s2)
        r_ad = self.lambda_ad * self.q_ad.get((q, q2), 0.0) if q != q2 else 0.0       # Eq. 6
        g_pd = 0.0
        pi_t = self.policy.get(q)
        if pi_t is not None and self.lambda_pd:
            g_pd = self.lambda_pd * (pi_t[a] - softmax(self.q[s])[a])                 # Eq. 7
        self.q[s][a] += self.cfg.alpha * (omega * delta + (1.0 - omega) * (r_ad + g_pd))   # Eqs. 10-11


METHODS = {
    "no_transfer": NoTransfer,
    "ad": AutomatonDistillation,
    "pd": PolicyDistillation,
    "kickstarting": Kickstarting,
    "cadent": CADENT,
}

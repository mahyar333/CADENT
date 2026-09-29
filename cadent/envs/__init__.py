"""Teacher (source) and student (target) versions of the Blind Craftsman benchmark.

The layouts are those of the paper's experiments: the teacher learns a small version of the
task, and the student learns a larger version with a different layout (domain shift).
"""

from __future__ import annotations

from .base import ProductEnv
from .gridworlds import BlindCraftsman

ENVIRONMENTS = ("blind_craftsman",)


def make(name: str, role: str = "student", sparse: bool = False) -> ProductEnv:
    """Build the ``teacher`` (source) or ``student`` (target) environment of a benchmark.

    ``sparse=True`` removes the automaton-progress rewards, so only task completion is rewarded.
    Both roles and all methods then see the same sparse reward.
    """
    teacher = role == "teacher"
    if name != "blind_craftsman":
        raise ValueError(f"unknown environment {name!r}; choose from {ENVIRONMENTS}")
    if teacher:
        env = BlindCraftsman(6, {(1, 1): "wood", (3, 4): "factory", (5, 5): "home"},
                             obstacles=[(2, 2)], max_steps=250, name=name)
    else:
        env = BlindCraftsman(25, {(2, 11): "wood", (15, 3): "wood", (21, 21): "wood",
                                  (8, 20): "factory", (24, 12): "home"}, max_steps=2500, name=name)
    env.progress_rewards = not sparse
    return env


__all__ = ["ENVIRONMENTS", "ProductEnv", "make", "BlindCraftsman"]

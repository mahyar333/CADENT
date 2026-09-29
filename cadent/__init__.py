"""CADENT: Context-Aware Distillation with Experience-gated Transfer (AISTATS 2026)."""

from .agents import CADENT, METHODS, AutomatonDistillation, Kickstarting, NoTransfer, PolicyDistillation
from .envs import ENVIRONMENTS, make
from .teacher import TeacherConfig, distill, teacher_knowledge

__version__ = "1.0.0"
__all__ = ["CADENT", "METHODS", "AutomatonDistillation", "Kickstarting", "NoTransfer", "PolicyDistillation",
           "ENVIRONMENTS", "make", "TeacherConfig", "distill", "teacher_knowledge"]

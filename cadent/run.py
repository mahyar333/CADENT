"""Run CADENT and the baselines on one benchmark.

    python -m cadent.run --config configs/blind_craftsman.yaml
    python -m cadent.run --config configs/blind_craftsman.yaml --methods cadent ad --seeds 0 1 2
    python -m cadent.run --config configs/blind_craftsman.yaml --ablation

Every (method, seed) run trains the seed's teacher on the source task, distils its knowledge, and
trains the student on the target task. Results are written as JSON to ``results/<env>/<method>/``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from .agents import CADENT, METHODS, QLearningConfig
from .envs import make
from .teacher import TeacherConfig, teacher_knowledge

ABLATIONS = {
    "cadent": {},
    "no_trust_gate": {"trust": "fixed", "fixed_trust": 0.5, "name": "No Trust Gate"},
    "ad_only": {"lambda_pd": 0.0, "name": "AD Only"},
    "pd_only": {"lambda_ad": 0.0, "strategic": "reward", "name": "PD Only"},
}
LABELS = {"no_transfer": "No Transfer", "ad": "Automaton Distillation", "pd": "Policy Distillation",
          "kickstarting": "Kickstarting", "cadent": "CADENT", "no_trust_gate": "No Trust Gate",
          "ad_only": "AD Only", "pd_only": "PD Only"}


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_one(cfg: Dict, method: str, seed: int) -> Dict:
    """Train the teacher and one student for one seed; returns the logs."""
    sparse = cfg.get("sparse_rewards", False)
    teacher_env, student_env = make(cfg["env"], "teacher", sparse), make(cfg["env"], "student", sparse)
    knowledge = teacher_knowledge(teacher_env, TeacherConfig(**cfg["teacher"]),
                                  np.random.default_rng([seed, 0]))
    rng = np.random.default_rng([seed, 1])
    ql = QLearningConfig(**cfg["q_learning"])
    base = "cadent" if method in ABLATIONS else method
    params = dict(cfg["methods"].get(base, {}))
    tau = params.pop("tau", None)                      # teacher-policy temperature of Eq. 3
    if tau is not None:
        knowledge = knowledge.with_temperature(teacher_env, tau)
    if method in ABLATIONS:
        params.update(ABLATIONS[method])
        agent = CADENT(student_env, ql, rng, knowledge, **params)
    else:
        if method == "kickstarting":
            params["total_episodes"] = cfg["episodes"]
        cls = METHODS[method]
        agent = cls(student_env, ql, rng) if method == "no_transfer" else cls(student_env, ql, rng, knowledge, **params)
    log = agent.train(cfg["episodes"])
    return {"env": cfg["env"], "method": method, "seed": seed, "teacher_success": knowledge.success_rate,
            "q_ad_transitions": len([t for t in knowledge.q_ad if t[0] != t[1]]), **asdict(log)}


def _job(args):
    cfg, method, seed, out = args
    path = Path(out) / cfg["env"] / method / f"seed{seed}.json"
    if path.exists():
        return str(path)
    result = run_one(cfg, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return str(path)


def run(cfg: Dict, methods: List[str], seeds: List[int], out: str = "results", workers: int = 0) -> None:
    jobs = [(cfg, m, s, out) for m in methods for s in seeds]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with Pool(min(workers, len(jobs))) as pool:
        for i, path in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--ablation", action="store_true", help="run the CADENT ablation variants")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    methods = args.methods or (list(ABLATIONS) if args.ablation else list(METHODS))
    run(cfg, methods, args.seeds or cfg["seeds"], args.out, args.workers)


if __name__ == "__main__":
    main()

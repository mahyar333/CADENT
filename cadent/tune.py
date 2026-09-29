"""Hyperparameter search with an equal budget for every transfer method.

    python -m cadent.tune --config configs/blind_craftsman.yaml --write configs/private/blind_craftsman.yaml

All methods share the Q-learning settings of the config (step size, discount, exploration schedule).
Only each method's own parameters are searched: the full grid when it has at most ``budget``
points, otherwise ``budget`` configurations drawn at random from it. Candidates are scored on
tuning seeds that are disjoint from the evaluation seeds, by the mean episode reward over the
whole run (the area under the learning curve, which rewards both speed and final performance).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import signal
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from .run import load_config, run_one

SEARCH_SPACE = {
    "ad": {"rho": [0.99, 0.995, 0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99999]},
    "pd": {"beta": [0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8], "temperature": [1.0, 5.0, 20.0, 50.0, 100.0],
           "tau": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]},
    "kickstarting": {"beta": [0.05, 0.1, 0.2, 0.4, 0.8, 1.6], "temperature": [1.0, 5.0, 20.0, 50.0, 100.0],
                     "tau": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0], "anneal_fraction": [0.25, 0.5, 1.0]},
    "cadent": {"rho": [0.9, 0.999, 0.9995, 0.9998, 0.9999], "lambda_ad": [0.0, 0.3, 1.0, 4.0],
               "lambda_pd": [0.05, 0.1, 0.3, 0.6], "theta": [0.25, 0.5, 1.0, 2.0, 4.0], "eta": [0.01, 0.1],
               "k": [5.0], "tau": [1.0, 2.0, 5.0]},
}
TUNING_SEEDS = [100, 101, 102, 103, 104]


def candidates(method: str, budget: int, rng: np.random.Generator):
    space = SEARCH_SPACE[method]
    grid = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    if len(grid) <= budget:
        return grid
    return [grid[i] for i in rng.choice(len(grid), size=budget, replace=False)]


TIME_LIMIT = 300      # seconds per tuning run; a run that exceeds it counts as a failed configuration


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _score(args):
    cfg, method, params, seed = args
    cfg = copy.deepcopy(cfg)
    cfg["methods"][method] = dict(cfg["methods"].get(method, {}), **params)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIME_LIMIT)
    try:
        return float(np.mean(run_one(cfg, method, seed)["rewards"]))
    except _Timeout:
        return float("-inf")
    finally:
        signal.alarm(0)


def tune(cfg, methods, budget=60, workers=9, seed=0):
    rng = np.random.default_rng(seed)
    report = {}
    with Pool(workers) as pool:
        for method in methods:
            cands = candidates(method, budget, rng)
            jobs = [(cfg, method, p, s) for p in cands for s in TUNING_SEEDS]
            scores = np.array(pool.map(_score, jobs)).reshape(len(cands), len(TUNING_SEEDS)).mean(axis=1)
            order = np.argsort(-scores)
            report[method] = [{"params": cands[i], "score": float(scores[i])} for i in order]
            print(f"{cfg['env']} {method}: best {scores[order[0]]:.2f} with {cands[order[0]]} "
                  f"({len(cands)} candidates)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="+", default=list(SEARCH_SPACE))
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--report", default=None, help="where to save the full ranking (JSON)")
    parser.add_argument("--write", default=None, help="config file to update with the best values")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = tune(cfg, args.methods, args.budget, args.workers)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1))
    if args.write:
        for method, ranking in report.items():
            cfg["methods"][method] = dict(cfg["methods"].get(method, {}), **ranking[0]["params"])
        with open(args.write, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()

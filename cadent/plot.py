"""Figures and summary tables from the JSON results.

    python -m cadent.plot --env blind_craftsman
    python -m cadent.plot --env blind_craftsman --ablation

For each benchmark: reward per episode, steps per episode, and reward against cumulative
environment steps (mean over seeds, shaded standard error), plus a Markdown table with the final
reward and the environment steps needed to reach 90% of the best final reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .run import ABLATIONS, LABELS

COLORS = {"no_transfer": "#7f7f7f", "ad": "#2a78d6", "pd": "#e08a1e", "kickstarting": "#8e5bb5",
          "cadent": "#1a9b5a", "no_trust_gate": "#d1495b", "ad_only": "#2a78d6", "pd_only": "#e08a1e"}
STYLES = {"no_transfer": ":", "no_trust_gate": "--", "ad_only": "-.", "pd_only": "-."}
TITLES = {"blind_craftsman": "Blind Craftsman"}
MAIN = ["no_transfer", "pd", "kickstarting", "ad", "cadent"]
FINAL, WINDOW = 100, 50

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})


def load(results: str, env: str, methods: List[str]) -> Dict[str, List[dict]]:
    out = {}
    for m in methods:
        runs = [json.loads(p.read_text()) for p in sorted((Path(results) / env / m).glob("seed*.json"))]
        if runs:
            out[m] = runs
    return out


def smooth(x: np.ndarray, w: int = WINDOW) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="valid")


def mean_se(rows: np.ndarray):
    m = rows.mean(0)
    se = rows.std(0, ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else np.zeros_like(m)
    return m, se


def samples_to_threshold(run: dict, threshold: float) -> float:
    r, cum = smooth(np.array(run["rewards"])), np.cumsum(run["steps"])[WINDOW - 1:]
    hit = np.flatnonzero(r >= threshold)
    return float(cum[hit[0]]) if len(hit) else float(np.sum(run["steps"]))


def summary(data: Dict[str, List[dict]]) -> List[dict]:
    finals = {m: np.array([np.mean(r["rewards"][-FINAL:]) for r in runs]) for m, runs in data.items()}
    start = np.mean([np.mean(r["rewards"][:WINDOW]) for runs in data.values() for r in runs])
    best = max(f.mean() for f in finals.values())
    threshold = start + 0.9 * (best - start)
    rows, base = [], None
    for m, runs in data.items():
        samples = np.array([samples_to_threshold(r, threshold) for r in runs])
        reached = sum(smooth(np.array(r["rewards"])).max() >= threshold for r in runs)
        row = {"method": m, "final": finals[m], "samples": samples, "reached": reached, "n": len(runs),
               "auc": np.array([np.mean(r["rewards"]) for r in runs])}
        if m == "no_transfer":
            base = row
        rows.append(row)
    for row in rows:
        if base is not None and row is not base:
            row["gain"] = 100.0 * (1.0 - row["samples"].mean() / base["samples"].mean())
            row["p_samples"] = stats.mannwhitneyu(row["samples"], base["samples"], alternative="less").pvalue
    return rows, threshold


def table(rows: List[dict], threshold: float, env: str) -> str:
    lines = [f"**{TITLES.get(env, env)}** (threshold: {threshold:.1f} reward, 90% of the way from the initial "
             f"to the best final reward)", "",
             "| Method | Final reward | Env. steps to threshold | Runs reaching it | Fewer steps than No Transfer |",
             "|---|:---:|:---:|:---:|:---:|"]
    for r in rows:
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        gain = "" if "gain" not in r else f"{r['gain']:+.0f}% (p={r['p_samples']:.2g})"
        lines.append(f"| {LABELS[r['method']]} | {r['final'].mean():.1f} ± {se(r['final']):.1f} | "
                     f"{r['samples'].mean() / 1e3:.0f}k ± {se(r['samples']) / 1e3:.0f}k | {r['reached']}/{r['n']} | {gain} |")
    return "\n".join(lines)


def figures(data: Dict[str, List[dict]], env: str, out: Path, prefix: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    specs = [("reward_per_episode", "Episode", "Reward"), ("steps_per_episode", "Episode", "Steps per episode"),
             ("reward_per_cumulative_steps", "Environment steps (millions)", "Reward")]
    for fname, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for m, runs in data.items():
            key = "steps" if fname == "steps_per_episode" else "rewards"
            rows = np.array([smooth(np.array(r[key], float)) for r in runs])
            mu, se = mean_se(rows)
            if fname == "reward_per_cumulative_steps":
                x = np.mean([np.cumsum(r["steps"])[WINDOW - 1:] for r in runs], axis=0) / 1e6
            else:
                x = np.arange(WINDOW, WINDOW + len(mu))
            lw = 2.0 if m == "cadent" else 1.3
            ax.plot(x, mu, color=COLORS[m], ls=STYLES.get(m, "-"), lw=lw, label=LABELS[m])
            ax.fill_between(x, mu - se, mu + se, color=COLORS[m], alpha=0.15, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(TITLES.get(env, env))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{prefix}{env}_{fname}.png", dpi=200)
        plt.close(fig)


def trust_figure(data: Dict[str, List[dict]], env: str, out: Path) -> None:
    if "cadent" not in data:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    rows = np.array([smooth(np.nan_to_num(np.array(r["trust"], float), nan=1.0)) for r in data["cadent"]])
    mu, se = mean_se(rows)
    x = np.arange(WINDOW, WINDOW + len(mu))
    ax.plot(x, mu, color=COLORS["cadent"], lw=1.8)
    ax.fill_between(x, mu - se, mu + se, color=COLORS["cadent"], alpha=0.15, lw=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Mean trust $\omega$")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{TITLES.get(env, env)}: trust dynamics")
    fig.tight_layout()
    fig.savefig(out / f"{env}_trust.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="assets/results")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    methods = list(ABLATIONS) if args.ablation else MAIN
    data = load(args.results, args.env, methods)
    out = Path(args.out)
    figures(data, args.env, out, "ablation_" if args.ablation else "")
    if not args.ablation:
        trust_figure(data, args.env, out)
    rows, threshold = summary(data)
    text = table(rows, threshold, args.env)
    (out / f"{'ablation_' if args.ablation else ''}{args.env}_summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

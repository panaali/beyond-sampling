"""Ladder table (markdown) from results_H{H}_T3.json for the frozen recipe and the reference tiers.
usage: python ladder.py [tier]   (default: the frozen recipe run_t3.RECIPE)"""
import json
import sys

import numpy as np

import run_t3
from zoo import ops_per_sample


def per_model_ratio(r, tier, ops1):
    p = r["truth"]; e = r[tier]["est"] - p
    return e * e / (max(p * (1 - p), 1e-12) / (r[tier]["ops"] / ops1))


def main(tier):
    print(f"| model | params | linear regions (mean, seeds) | exact-enum ops | {tier} ops | {tier} traces | "
          f"MSE ratio: seeds / ARC abs 0.05 / rel 0.0025 / 0.005 / 0.01 / 0.02 | seeds passed (per model) | "
          f"worst single model |")
    print("|---|---|---|---|---|---|---|---|---|")
    for H in (4, 5, 6, 8):
        try:
            R = json.load(open(f"results_H{H}_T3.json"))
        except FileNotFoundError:
            continue
        rows = R["rows"]
        if tier not in next(iter(rows.values())):
            continue
        run_t3.TIERS = (tier,)
        S = run_t3.summary(H, rows)
        ops1 = ops_per_sample(H, 3)
        seeds = [rows[f"seed{s}"] for s in range(5)]
        npass = sum(per_model_ratio(r, tier, ops1) < 1 for r in seeds)
        worst = max(per_model_ratio(r, tier, ops1) for r in rows.values())
        params = H + H * H + 3 * H
        g = ["seed", "abs0.05", "s0.0025", "s0.005", "s0.01", "s0.02"]
        print(f"| M_{H},3 | {params} | {np.mean([r['regions'] for r in seeds]):.0f} | {S['seed']['exact_ops']:.0f} | "
              f"{S['seed'][tier]['ops']:.0f} | {S['seed'][tier]['traces']:.0f} | "
              + " / ".join(f"{S[x][tier]['mse_ratio']:.1e}" for x in g) + f" | {npass}/5 | {worst:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else run_t3.RECIPE)

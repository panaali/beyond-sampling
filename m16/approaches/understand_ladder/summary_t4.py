"""Tables for T = 4 rungs from results_H{H}_T4.json (run_t4.py). MSE vs truth with the MC truth's variance subtracted
(leaderboard convention: MSE = mean(e^2) - mean(se^2)); sampling MSE at equal ops = mean p(1-p) / (ops / ops_per_sample);
per-model ratio = max(e^2 - se^2, 0) / (p(1-p) / (ops / ops_per_sample)). usage: python summary_t4.py H [tier]"""
import json
import sys

import numpy as np

from zoo import ops_per_sample


def group(n):
    return "seed" if n.startswith("seed") else n.rsplit("_", 1)[0].replace("_n0", "")


def main(H, tier):
    R = json.load(open(f"results_H{H}_T4.json"))["rows"]
    rows = {k: v for k, v in R.items() if tier in v}
    ops1 = ops_per_sample(H, 4)
    print(f"M_{H},4 tier {tier}: {len(rows)} models")
    print("| group | n | mean acc | patterns (>= regions) | face cells | ops | rmse | bias | MSE ratio | worst per-model |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in sorted(set(group(n) for n in rows)):
        ns = [n for n in rows if group(n) == g]
        tr = np.array([rows[n]["truth"] for n in ns]); se = np.array([rows[n]["truth_se"] for n in ns])
        e = np.array([rows[n][tier]["est"] for n in ns]) - tr
        ops = np.array([rows[n][tier]["ops"] for n in ns]); pq = tr * (1 - tr)
        mse = max(np.mean(e ** 2) - np.mean(se ** 2), 0.0)
        ratio = mse / (np.mean(pq) / (np.mean(ops) / ops1))
        pm = np.maximum(e ** 2 - se ** 2, 0) / (pq / (ops / ops1))
        print(f"| {g} | {len(ns)} | {tr.mean():.4f} | {np.mean([rows[n]['patterns'] for n in ns]):.0f} | "
              f"{np.mean([rows[n]['face_cells'] for n in ns]):.0f} | {np.mean(ops) / 1e6:.0f}M | {np.sqrt(np.mean(e ** 2)):.1e} | "
              f"{e.mean():+.1e} | {ratio:.2e} | {pm.max():.2f} |")
    for n in sorted(rows):
        if n.startswith("seed"):
            r = rows[n]
            print(f"   {n}: truth {r['truth']:.6f} +- {r['truth_se']:.1e}, est {r[tier]['est']:.6f}, err {r[tier]['est'] - r['truth']:+.1e}, "
                  f"ops {r[tier]['ops'] / 1e6:.0f}M, traces {r[tier]['traces']}, patterns {r['patterns']}, face cells {r['face_cells']}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "F2L3")

"""Tables from results_H{H}_T3.json (written by run_t3.py). usage: python report.py H [H ...]"""
import json
import sys

import numpy as np

from run_t3 import summary

for H in map(int, sys.argv[1:]):
    R = json.load(open(f"results_H{H}_T3.json"))
    rows = R["rows"]
    tiers = [t for t in ("F0", "F1a", "F1", "F2", "F2L2", "FX") if t in next(iter(rows.values()))]
    import run_t3
    run_t3.TIERS = tuple(tiers)
    S = summary(H, rows)
    print(f"== M_{H},3 ==")
    for g, d in S.items():
        print(f"{g:10s} n={d['n']:3d} acc {d['mean_truth']:.4f} regions {d['regions']:.0f} exact_ops {d['exact_ops']:.0f}")
        for t in tiers:
            x = d[t]
            print(f"   {t:4s} rmse {x['rmse']:.2e} bias {x['bias']:+.2e} max {x['max_abs']:.2e} ops {x['ops']:.0f} "
                  f"traces {x['traces']:.0f} MSE ratio {x['mse_ratio']:.2e} (worst model {x['per_model_ratio_max']:.2e})")
    print("  per seed (est - truth):")
    for s in range(5):
        x = rows[f"seed{s}"]
        print(f"   seed{s} truth {x['truth']:.5f} regions {x['regions']} " + " ".join(f"{t} {x[t]['est'] - x['truth']:+.1e}" for t in tiers))

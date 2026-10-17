"""Comparison for rung T88: the T75 recipe (understand_m43/est75.py, imported unchanged) on the same models and
truths as run_t3.py (results_H{H}_T3.json). The frozen RECIPE has K = (4, 4) (= 'no cap' at H = 4); at H > 4 we
also run K = (H, H), the literal port of 'no cap'. usage: python run_t75.py H"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import est75  # noqa: E402
from m43 import Ops  # noqa: E402
from run_t3 import group  # noqa: E402
from zoo import best_seed, leaderboard_models, noisy_abs, ops_per_sample, zoo_model  # noqa: E402

if __name__ == "__main__":
    H = int(sys.argv[1]); T = 3
    R = json.load(open(os.path.join(HERE, f"results_H{H}_T3.json")))
    K = R["K"]
    ms = leaderboard_models(H, T)
    base = zoo_model(H, T, best_seed(H, T))
    for ns in (0, 1):
        for k, m in enumerate(noisy_abs(base, K, 0.05, ns)):
            ms[f"abs0.05_n{ns}_{k}"] = m
    variants = {"T75 frozen K=(4,4)": dict(est75.RECIPE), f"T75 K=({H},{H})": dict(est75.RECIPE, K=(H, H))}
    out = {}
    ops1 = ops_per_sample(H, T)
    for vn, kw in variants.items():
        rows = {}
        for name, m in ms.items():
            o = Ops(); e = est75.est75(m, o, **kw)
            rows[name] = {"est": e, "ops": o.n, "truth": R["rows"][name]["truth"]}
        out[vn] = rows
        print(f"== {vn} on M_{H},3")
        for g in sorted(set(group(n) for n in rows)):
            names = [n for n in rows if group(n) == g]
            tr = np.array([rows[n]["truth"] for n in names]); e = np.array([rows[n]["est"] for n in names]) - tr
            ops = np.mean([rows[n]["ops"] for n in names]); pq = np.mean(tr * (1 - tr))
            print(f"  {g:10s} rmse {np.sqrt(np.mean(e**2)):.2e} bias {e.mean():+.2e} ops {ops:.0f} "
                  f"MSE ratio {np.mean(e**2) / (pq / (ops / ops1)):.2e}")
        print("  seeds:", " ".join(f"{rows[f'seed{s}']['est'] - rows[f'seed{s}']['truth']:+.1e}" for s in range(5)))
    json.dump(out, open(os.path.join(HERE, f"results_t75_H{H}.json"), "w"), indent=1)

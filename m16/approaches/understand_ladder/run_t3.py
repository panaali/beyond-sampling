"""T73 rung scoring for T = 3 models (M_5,3, M_6,3, M_8,3; M_4,3 for reference).
Models: zoo seeds 0-4; ARC protocol (absolute sigma 0.05 on every weight, copies of the best-of-5 seed, noise seeds
0-1 x K copies); leaderboard protocol (relative sigma 0.0025-0.02 x matrix RMS, 6 copies each, = leaderboard.models).
Truth: exact enumeration (exact3; validated against 1e6-sample MC and against tier FX). Ops: multiply-add = 1
(ARC / T63 convention); sampling MSE at equal ops = mean p(1-p) / (mean ops / ops_per_sample).
usage: OMP_NUM_THREADS=1 python run_t3.py H [K]   -> results_H{H}_T3.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
from exact3 import count_regions, exact_acc  # noqa: E402
import m43  # noqa: E402
import fibre  # noqa: E402
from fibre import est_fibre, est_fibre2, Ops  # noqa: E402
from zoo import best_seed, leaderboard_models, noisy_abs, ops_per_sample, zoo_model  # noqa: E402

TIERS = ("F0", "F1a", "F1", "F2", "F2L2", "FX")
RECIPE = "F2L2"     # frozen ladder recipe (T88): attached wedges at both fibre ends, forward look-ahead of 2 pieces


def score(H, K, tiers=None, old=None):
    T = 3
    ms = leaderboard_models(H, T)
    base = zoo_model(H, T, best_seed(H, T))
    for ns in (0, 1):
        for k, m in enumerate(noisy_abs(base, K, 0.05, ns)):
            ms[f"abs0.05_n{ns}_{k}"] = m
    rows = {}
    t0 = time.time()
    for name, m in ms.items():
        if old is not None:
            r = old[name]
        else:
            oe = m43.Ops(); st = {}
            ex = exact_acc(m, oe, st)
            r = {"truth": ex, "exact_ops": oe.n, "leaves": st["leaves"], "regions": count_regions(m)}
        for tier in (tiers or TIERS):
            o = Ops(); inf = {}; tt = time.time()
            if tier in ("F2", "F2L2"):
                fibre.LOOK = 2 if tier == "F2L2" else 0
                e = est_fibre2(m, o, inf)
            else:
                e = est_fibre(m, tier, o, inf)
            r[tier] = {"est": e, "ops": o.n, "traces": inf["traces"], "segments": inf["segments"], "sec": time.time() - tt}
        rows[name] = r
    print(f"H={H}: {len(rows)} models in {time.time() - t0:.0f} s")
    return rows


def group(name):
    if name.startswith("seed"):
        return "seed"
    return name.rsplit("_", 1)[0].replace("_n0", "").replace("_n1", "")


def summary(H, rows, T=3):
    ops1 = ops_per_sample(H, T)
    out = {}
    for g in sorted(set(group(n) for n in rows)):
        names = [n for n in rows if group(n) == g]
        tr = np.array([rows[n]["truth"] for n in names]); pq = float(np.mean(tr * (1 - tr)))
        d = {"n": len(names), "mean_truth": float(tr.mean()), "pq": pq,
             "exact_ops": float(np.mean([rows[n]["exact_ops"] for n in names])),
             "regions": float(np.mean([rows[n]["regions"] for n in names]))}
        for tier in [t for t in TIERS if t in next(iter(rows.values()))]:
            e = np.array([rows[n][tier]["est"] for n in names]) - tr
            ops = float(np.mean([rows[n][tier]["ops"] for n in names]))
            mse_s = pq / (ops / ops1)
            d[tier] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()), "max_abs": float(np.abs(e).max()),
                       "ops": ops, "traces": float(np.mean([rows[n][tier]["traces"] for n in names])),
                       "sampling_rmse": float(np.sqrt(mse_s)), "mse_ratio": float(np.mean(e ** 2) / mse_s),
                       "per_model_ratio_max": float(max((rows[n][tier]["est"] - rows[n]["truth"]) ** 2 /
                                                         (max(rows[n]["truth"] * (1 - rows[n]["truth"]), 1e-12) /
                                                          (rows[n][tier]["ops"] / ops1)) for n in names))}
        out[g] = d
    return out


if __name__ == "__main__":
    H = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    tiers = sys.argv[3].split(",") if len(sys.argv) > 3 else None       # e.g. F2L2: add tiers to existing results
    path = os.path.join(HERE, f"results_H{H}_T3.json")
    old = json.load(open(path))["rows"] if tiers else None
    rows = score(H, K, tiers, old)
    summ = summary(H, rows)
    json.dump({"H": H, "T": 3, "K": K, "rows": rows, "summary": summ}, open(path, "w"), indent=1)
    for g, d in summ.items():
        print(f"{g:10s} n={d['n']:3d} acc {d['mean_truth']:.4f} regions {d['regions']:.0f} exact_ops {d['exact_ops']:.0f}")
        for tier in TIERS:
            if tier not in d:
                continue
            x = d[tier]
            print(f"   {tier:4s} rmse {x['rmse']:.2e} bias {x['bias']:+.2e} max {x['max_abs']:.2e} ops {x['ops']:.0f} "
                  f"samp_rmse {x['sampling_rmse']:.2e} MSE ratio {x['mse_ratio']:.2e} (worst model {x['per_model_ratio_max']:.2e})")

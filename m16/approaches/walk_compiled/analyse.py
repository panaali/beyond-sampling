"""Equal-MSE wall-time ratios from bench_T5.json / bench_T10.json -> analysis.json + printed table.
R = t_walk / t_samp(n_eq), n_eq = mean p(1-p) / MSE_walk, t_samp(n) = n x (sampler s/FPE at 2^20 inputs)."""
import json

import numpy as np

import models as Mo

PAIRS = [("fast", "compiled_rng", "pre-registered: compiled walk, face maps timed  vs  compiled sampler, RNG timed"),
         ("fast_pre", "compiled_pre", "model-free part precomputed on both sides"),
         ("fastfm", "compiled_rng", "exploratory: reassociation/FMA walk vs compiled sampler, RNG timed"),
         ("fastfm_pre", "compiledfm_pre", "exploratory: both fastmath, precomputed"),
         ("repro", "compiled_rng", "scipy CDF walk vs compiled sampler"),
         ("numpy", "numpy", "frozen numpy walk vs numpy sampler (T379's setting)")]
GROUPS = {5: {"zoo": [f"z{s}" for s in range(5)], "fresh": [f"f{s}" for s in range(100, 105)]},
          10: {"zoo": [f"z{s}" for s in range(5)]}}
N = 1 << 20
res = {}
for T in (5, 10):
    try:
        b = json.load(open(f"{Mo.HERE}/bench_T{T}.json"))
    except FileNotFoundError:
        continue
    for g, names in GROUPS[T].items():
        names = [n for n in names if n in b["models"]]
        if not names:
            continue
        ms = [b["models"][n] for n in names]
        p = np.array([m["truth"]["acc"] for m in ms]); se = np.array([m["truth"]["se"] for m in ms])
        for B in ("16000", "64000"):
            est = np.array([m["est"][B]["fast"]["est"] for m in ms])
            cost = np.mean([m["est"][B]["fast"]["cost"] for m in ms])
            mse_w = float(np.mean((est - p) ** 2) - np.mean(se ** 2))
            mse_s = float(np.mean(p * (1 - p)) / cost)
            n_eq = float(np.mean(p * (1 - p)) / max(mse_w, 1e-300))
            row = {"models": names, "mse_walk": mse_w, "mse_samp_same_cost": mse_s, "mse_ratio": mse_w / mse_s,
                   "cost": cost, "n_eq": n_eq, "pairs": {}}
            for w, s, desc in PAIRS:
                tw = float(np.mean([m["est"][B][w]["t"] for m in ms]))
                tww = float(np.mean([m["est"][B][w]["t_walk"] for m in ms]))
                rate = float(np.mean([m["sampler"][s][str(N)] for m in ms]) / N)
                t_same = float(np.mean([m["sampler"][s][B] for m in ms]))
                row["pairs"][f"{w}|{s}"] = {"desc": desc, "t_walk_est": tw, "t_in_walk": tww,
                                            "samp_s_per_fpe": rate, "t_samp_same_budget": t_same,
                                            "time_ratio_same_fpe": tw / (cost * rate),
                                            "R_equal_mse": tw / (n_eq * rate)}
            res[f"T{T}:{g}:{B}"] = row
            print(f"T{T} {g:5s} {B:>5s}  MSE ratio {row['mse_ratio']:.3f}  n_eq {n_eq:,.0f}")
            for k, v in row["pairs"].items():
                print(f"    {k:28s} t_est {v['t_walk_est'] * 1e3:7.1f} ms (in walk {v['t_in_walk'] * 1e3:6.1f})"
                      f"  samp {v['samp_s_per_fpe'] * 1e9:6.0f} ns/FPE  same-FPE ratio {v['time_ratio_same_fpe']:5.2f}"
                      f"  R(equal MSE) {v['R_equal_mse']:.3f}")
json.dump(res, open(f"{Mo.HERE}/analysis.json", "w"), indent=1)

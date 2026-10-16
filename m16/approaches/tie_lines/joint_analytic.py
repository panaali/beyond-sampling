"""Precise expected compute-adjusted gain of est_joint (and of stratified.py, strat_swap) per model, from 1e6-draw
per-stratum variances (independent sample, seed 31; design pilot used seed 99). Includes rounding (n_h >= 1) and
input-generation cost. gain = acc(1-acc) / (Var(B) * cost(B)).   usage: python joint_analytic.py"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import estimators as E
import joint as J
from stratified import stratum
from arc_eval import noisy_models
from common import Wi, Wh, Wo
D = J._design(); P = D["P"]; G = len(P); use = D["use_swap"].astype(bool)
x = np.random.default_rng(31).standard_normal((1_000_000, 10)); h = J.jstratum(x); h7 = stratum(x)
y = E.swap_minpair(x)[0]
models = {"orig": (Wi, Wh, Wo)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s)):
        models[f"s{s}_{k}"] = m
BUD = (1000, 4000, 16000, 64000)
def sd(lab, v, K):
    n = np.bincount(lab, minlength=K); m1 = np.bincount(lab, v, K) / n; m2 = np.bincount(lab, v * v, K) / n
    return np.sqrt(np.maximum(m2 - m1 ** 2, 0))
rows = []
for name, m in models.items():
    g = E.g_of(m, x); sw = (g + E.g_of(m, y)) / 2; acc = g.mean(); ref = acc * (1 - acc)
    sig = np.where(use, sd(h, sw, G), sd(h, g, G)); c = np.where(use, 2.0, 1.0)
    r = {"model": name, "acc": round(float(acc), 4)}
    for B in BUD:
        Bp = B / (1 + D["ovh"]); n = np.maximum(1, np.round(D["alloc"] * Bp))
        var = (P ** 2 * sig ** 2 / n).sum(); cost = (n * c).sum() + J.GEN_COST * 1.05 * (n / P).max()
        r[f"joint@{B}"] = round(float(ref / (var * cost)), 3)
    # baselines (large-budget limit): stratified.py, with and without its rejection cost
    s7 = sd(h7, g, 7); a0 = E.ALLOC0
    v7 = (E.P ** 2 * s7 ** 2 / a0).sum()
    r["strat"] = round(float(ref / v7), 3); r["strat_gen"] = round(float(ref / (v7 * (1 + J.GEN_COST * (a0 / E.P).sum()))), 3)
    s7s = np.where(E.USE_SWAP, sd(h7, sw, 7), s7)
    r["strat_swap"] = round(float(ref / ((E.P ** 2 * s7s ** 2 / E.SW_N).sum())), 3)
    rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(os.path.join(HERE, "joint_analytic.json"), "w"), indent=1)

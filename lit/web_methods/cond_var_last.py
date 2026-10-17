"""Companion to cond_var.py: Var(E[f | last k inputs]) / p(1-p) for k=1..6 (n=2000 conditions x 300 inner draws, seed 2).
Usage: OMP_NUM_THREADS=1 python lit/web_methods/cond_var_last.py"""
import sys, numpy as np
sys.path.insert(0, 'm16')
from common import Wi, Wh, Wo, T
n, m = 2000, 300; rng = np.random.default_rng(2)
for k in (1, 2, 3, 4, 5, 6):  # condition on the LAST k inputs, average over the first T-k
    S = rng.standard_normal((n, k)); P = rng.standard_normal((n, m, T - k))
    x = np.concatenate([P, np.repeat(S[:, None], m, 1)], 2)
    h = np.zeros((n, m, len(Wi)))
    for j in range(T):
        h = np.maximum(h @ Wh.T + x[:, :, j:j+1] * Wi, 0)
    F = ((h @ Wo.T).argmax(2) == np.argsort(x, 2)[:, :, -2]).astype(float)
    p = F.mean(); vg = F.mean(1).var(ddof=1) - F.var(1, ddof=1).mean() / m
    print(f"condition on last k={k}: Var(E[f|last k])/p(1-p) = {vg/(p*(1-p)):.3f}", flush=True)

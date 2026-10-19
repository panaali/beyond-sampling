"""T389 development: does a weights-only position-influence feature explain the per-model deviation of cell sd from
the DEV2 table? g_s = || Wo (Wh/2)^(T-1-s) Wi/2 || (half-active gating), features log g_a, log g_b, per (stratum)
least squares fitted on DEV2 models 0-19, tested on 20-39 (allocation gain over the table, as in d10_cv)."""
import numpy as np
from d10_cv import X, fit, arms
D = np.load("../../../data/dev2/weights.npz"); T = 10
def g(i):
    Wi, Wh, Wo = D["Wi"][i], D["Wh"][i], D["Wo"][i]
    return np.array([np.linalg.norm(Wo @ np.linalg.matrix_power(Wh / 2, T - 1 - s) @ (Wi / 2)) for s in range(T)])
G = np.log(np.array([g(i) for i in range(40)]))
a = np.repeat(np.arange(T), T); b = np.tile(np.arange(T), T)
def feats(i): return np.c_[G[i][a], G[i][b], np.ones(T * T)]
tab = fit(X[:20])
coef = []
for j in range(5):
    A, yv = [], []
    for i in range(20):
        w, s = X[i, j, :, 0], X[i, j, :, 2]; ok = (w > 0) & (s > 0)
        rel = s / (w * s).sum()
        A.append(feats(i)[ok]); yv.append(np.log(rel[ok]) - np.log(tab[j][ok]))
    A, yv = np.concatenate(A), np.concatenate(yv)
    cf, *_ = np.linalg.lstsq(A, yv, rcond=None); r2 = 1 - np.var(yv - A @ cf) / np.var(yv); coef.append(cf)
    print(f"stratum {j}: coef {np.round(cf, 3)}  in-sample R2 {r2:.3f}")
r = []
for i in range(20, 40):
    t2 = np.array([tab[j] * np.exp(feats(i) @ coef[j]) for j in range(5)])
    a0 = arms(X[i], tab); a1 = arms(X[i], t2); r.append(a0["table"] / a1["table"])
print(f"held-out gain of table+weights over table: median {np.median(r):.3f} [min {min(r):.3f}, max {max(r):.3f}]")

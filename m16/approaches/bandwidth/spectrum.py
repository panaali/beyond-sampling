"""Sensitivity spectrum of delta. In the linear piece at the tie, delta_lin = -m/s with m = Jab . y, so the exact
gradient with respect to the inputs is g = -Jab/s (tie coordinates merged: dv = -(Jab_a + Jab_b)/s).
Per (a,b,p1) class: (1) eigen-spectrum of C = E[g g^T] (active subspace): how many directions matter;
(2) gradient heterogeneity H = E|g - E g|^2 / E|g|^2 (0: delta linear in the class; 1: gradients random).
usage: python spectrum.py <model>"""
import sys
import numpy as np
nm = sys.argv[1] if len(sys.argv) > 1 else "orig"
R = dict(np.load(f"data/{nm}.npz")); k = R["v"] > 0
J, s, a, b, p1 = R["Jab"][k], R["s"][k], R["a"][k], R["b"][k], R["p1"][k]
n = len(s); r = np.arange(n)
g = -J / s[:, None]
gv = g[r, a] + g[r, b]
g[r, a] = gv; g[r, b] = 0.0                       # coordinate a now carries d delta/d v; b slot empty
g = np.clip(g, -1, 1)                             # guard s ~ 0
cls = (a * 10 + b) * 10 + p1
eff, H, top = [], [], []
for c in np.unique(cls):
    i = cls == c
    if i.sum() < 40:
        continue
    G = np.delete(g[i], b[i][0], axis=1)
    C = G.T @ G / len(G); ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    eff.append(ev.sum() ** 2 / (ev ** 2).sum()); top.append(ev[:3] / ev.sum())
    H.append(((G - G.mean(0)) ** 2).sum(1).mean() / (G ** 2).sum(1).mean())
top = np.array(top)
print(f"{nm}: classes {len(H)}; participation ratio of grad-covariance spectrum (out of 9): median {np.median(eff):.2f};"
      f" top-1/2/3 eigen shares {top.mean(0).round(2)}; gradient heterogeneity H median {np.median(H):.2f} (p10 {np.percentile(H,10):.2f}, p90 {np.percentile(H,90):.2f})")

"""Diagnosis: smoothness of the bath-averaged segment mass E[F | config, v, o1] on a grid of (v, o1) quantiles.
usage: python diag_top.py <model> a b p1 [nb]"""
import sys
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
import wtool as W

nm = sys.argv[1]; a, b, p1 = map(int, sys.argv[2:5]); nb = int(sys.argv[5]) if len(sys.argv) > 5 else 256
M = W.get_model(nm)
bath = [k for k in range(10) if k not in (a, b, p1)]
Ub = qmc.Sobol(7, scramble=True, seed=1).random(nb)
G = 8
qv = (np.arange(G) + 0.5) / G
print(f"{nm} config a={a} b={b} p1={p1}: rows = v quantile, cols = o1 quantile (conditional on v); entries E[F]*1e3")
for uv in qv:
    v = W.inv_ybar(np.array([uv]))[0]
    row = []
    for uo in qv:
        o1 = ndtri(ndtr(v) + uo * (1 - ndtr(v)))
        lo = ndtri(np.clip(Ub * ndtr(v), 1e-300, 1 - 1e-16))
        y = np.empty((nb, 10)); y[:, a] = v; y[:, b] = v; y[:, p1] = o1; y[:, bath] = lo
        tau = W.SQ2 * np.minimum(o1 - v, v - lo.max(1))
        F = W.walk(M, y, np.full(nb, a), np.full(nb, b), tau)["mass"]
        row.append(F.mean() * 1e3)
    print(f" v={v:+.2f}: " + " ".join(f"{r:6.2f}" for r in row), flush=True)

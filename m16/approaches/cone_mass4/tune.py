"""Tune (KAPPA_GOOD, TOL): max error vs a 64-node evaluation and mean ops, on 4000 cones of 4 kinds (as calibrate.py).
usage: python tune.py"""
import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import mass4 as M  # noqa: E402

rng = np.random.default_rng(11)
Cs = []
for t in range(4000):
    kind = t % 4
    if kind == 0:
        N = rng.standard_normal((4, 4))
    elif kind == 1:
        N = np.linalg.inv(rng.standard_normal((4, 4)).T)
    elif kind == 2:
        c = rng.standard_normal(4); N = np.linalg.inv((c + rng.uniform(0.05, 2) * rng.standard_normal((4, 4))).T)
    else:
        R = rng.standard_normal((4, 4)); R[:, 3] *= 10 ** rng.uniform(-3, 0); R[1] += 10 ** rng.uniform(-3, 0) * R[0]
        N = np.linalg.inv(R.T)
    C = M.corr_from_normals(N, M.Ops())
    Cs.append((C, M.mass_from_corr(C, M.Ops(), 64)))
out = []
for kg in (0.25, 0.5, 1.0, 1e9):
    for tol in (1e-8, 1e-9):
        M.KAPPA_GOOD, M.TOL = kg, tol
        E, O = [], []
        for C, ref in Cs:
            o = M.Ops(); E.append(abs(M.mass_from_corr(C, o) - ref)); O.append(o.n)
        out.append(dict(kappa_good=kg, tol=tol, max_err=float(max(E)), p999_err=float(np.quantile(E, 0.999)),
                        mean_ops=float(np.mean(O)), max_ops=int(max(O))))
        print(out[-1], flush=True)
json.dump(out, open(f"{D}/tune.json", "w"), indent=1)

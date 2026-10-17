"""Calibrate the node count against the pair-selection score: errors at n nodes vs a 64-node evaluation of the same
formula (itself checked against the 30-digit reference in test_mass4.py). usage: calibrate.py [N]"""
import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import mass4 as M  # noqa: E402

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = np.random.default_rng(7)
NS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
rows = []
for t in range(NC):
    kind = t % 4
    if kind == 0:
        N = rng.standard_normal((4, 4))
    elif kind == 1:
        N = np.linalg.inv(rng.standard_normal((4, 4)).T)
    elif kind == 2:                      # random rays in a cap of random width (estT-like simplices)
        c = rng.standard_normal(4); R = c + rng.uniform(0.05, 2) * rng.standard_normal((4, 4))
        N = np.linalg.inv(R.T)
    else:                                # moderately flat / close rays
        R = rng.standard_normal((4, 4)); R[:, 3] *= 10 ** rng.uniform(-3, 0); R[1] += 10 ** rng.uniform(-3, 0) * R[0]
        N = np.linalg.inv(R.T)
    C = M.corr_from_normals(N, M.Ops())
    ref, info = M.mass_from_corr(C, M.Ops(), 64, True)
    errs = [abs(M.mass_from_corr(C, M.Ops(), n) - ref) for n in NS]
    rows.append([info["score"], ref] + errs)
A = np.array(rows)
out = {"N": NC, "nodes": NS, "bins": []}
edges = [0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0001]
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (A[:, 0] >= lo) & (A[:, 0] < hi)
    if sel.sum() == 0:
        continue
    mx = A[sel, 2:].max(0)
    out["bins"].append(dict(lo=lo, hi=hi, count=int(sel.sum()), max_err={n: float(v) for n, v in zip(NS, mx)}))
    print(f"score [{lo},{hi}) n={sel.sum():5d} ", " ".join(f"{n}:{v:.0e}" for n, v in zip(NS, mx)), flush=True)
np.save(os.path.join(D, "..", "..", "..", "data", "cone_mass4_calib.npy"), A)
json.dump(out, open(f"{D}/calibrate.json", "w"), indent=1)

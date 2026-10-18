"""T280 H-E15: cells per unit mass by tie-distance stratum, for dd cells (data/cells_*.pkl).
Masses: rays -> simplicial cones -> robust-angle orthant probabilities (tie_tail_labels/prefix_mass.py, T222-validated).
usage: python he15.py TAG [TAG ...]   (TAG e.g. prefix_target_k4, final_zoo16_4_0) -> results/he15_TAG.json"""
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "tie_tail_labels"))
import prefix_mass as P  # noqa: E402

warnings.simplefilter("ignore")
EDGES = [0, 0.02, 0.05, 0.1, 0.2, np.inf]
LABELS = ["<.02", ".02-.05", ".05-.1", ".1-.2", ">=.2"]


def masses(cells, k):
    Cs, Ts, own = [], [], []
    for ci, (V, _) in enumerate(cells):
        for s in P.simplices(V, k):
            Rs = V[s].T
            if abs(np.linalg.det(Rs)) < 1e-14:
                continue
            N, TH = P.normals_angles(Rs); Cs.append(N @ N.T); Ts.append(TH); own.append(ci)
    C, TH, own = np.array(Cs), np.array(Ts), np.array(own)
    a = np.pi / 2 - TH
    if k == 3:
        p = 0.125 + (a[:, 0, 1] + a[:, 0, 2] + a[:, 1, 2]) / (4 * np.pi)
    elif k == 4:
        p = P.orth4(C, TH)[0]
    else:
        raise ValueError(k)
    return np.bincount(own, weights=p, minlength=len(cells))


def strata(cells, m):
    U = np.array([V.mean(0) / np.linalg.norm(V.mean(0)) for V, _ in cells])
    u = -np.sort(-U, 1)
    d = np.minimum(u[:, 0] - u[:, 1], u[:, 1] - u[:, 2]) / np.sqrt(2)
    blind = u[:, 1] <= 0
    out = {}
    for lo, hi, lab in zip(EDGES[:-1], EDGES[1:], LABELS):
        sel = (~blind) & (d >= lo) & (d < hi)
        out[lab] = dict(cells=int(sel.sum()), mass=float(m[sel].sum()))
    out["blind"] = dict(cells=int(blind.sum()), mass=float(m[blind].sum()))
    near = (~blind) & (d < 0.05); far = (~blind) & (d >= 0.1)
    dens = lambda s: s.sum() / max(m[s].sum(), 1e-300)
    for v in out.values():
        v["cells_per_mass"] = v["cells"] / max(v["mass"], 1e-300)
    return out, float(dens(near) / dens(far))


for tag in sys.argv[1:]:
    t0 = time.time()
    cells = pickle.load(open(os.path.join(HERE, "data", f"cells_{tag}.pkl"), "rb"))
    k = cells[0][0].shape[1]
    m = masses(cells, k)
    st, ratio = strata(cells, m)
    r = dict(tag=tag, cells=len(cells), mass_sum_minus_1=float(m.sum() - 1), min_mass=float(m.min()),
             strata=st, near_far=ratio, sec=time.time() - t0)
    json.dump(r, open(os.path.join(HERE, "results", f"he15_{tag}.json"), "w"), indent=1)
    print(tag, "cells", len(cells), "sum-1 %.1e" % r["mass_sum_minus_1"], "near/far %.1f" % ratio,
          {kk: round(v["cells_per_mass"]) for kk, v in st.items()}, round(r["sec"]), flush=True)

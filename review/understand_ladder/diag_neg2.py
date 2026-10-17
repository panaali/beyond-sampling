"""T97: locate the F2L2 over-count on the adversarial H=6 model. For family circle (0,1,2), s=-1, truth 1, compare
pointwise in theta: (a) F2L2's wrong-interval density from trace_F2 at that theta, (b) the density from the
author's FX trace, (c) the reviewer's brute-force density (dense u grid, own forward pass). usage: python diag_neg2.py"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder"))
import fibre  # noqa: E402
from fiber3 import FAMS  # noqa: E402

J = json.load(open(os.path.join(HERE, "adv_H6.json")))
m = tuple(np.array(w) for w in J["worst"]["model"])
Wi, Wh, Wo = m


def S(u):
    return u / math.sqrt(1 + u * u)


def brute(P, U, nn, truth, fcl, th, n=200001):
    c, s = math.cos(th), math.sin(th)
    ucl = fcl[0] * c + fcl[1] * s
    if ucl <= 0:
        return 0.0
    phi = np.linspace(0, math.atan(ucl), n)[1:]
    phm = 0.5 * (np.concatenate([[0], phi[:-1]]) + phi)
    x = (c * P + s * U)[None] + np.tan(phm)[:, None] * nn[None]
    h = np.zeros((len(x), len(Wi)))
    for t in range(3):
        h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
    wrong = (h @ Wo.T).argmax(1) != truth
    return float(np.sum(np.cos(phm) * wrong) * (phi[1] - phi[0]))   # int cos(phi) dphi over wrong set


def dens(iv, th):
    c, s = math.cos(th), math.sin(th)
    tot = 0.0
    for lo, hi in iv:
        uh = 0.0 if hi is None else hi[0] * c + hi[1] * s
        ul = 0.0 if lo is None else lo[0] * c + lo[1] * s
        tot += S(uh) - S(ul)
    return tot


fam = [f for f in FAMS if f["circle"] == (0, 1, 2) and f["s"] == -1][0]
P, U = fam["P"], fam["U"]
for nn, truth, wcl in fam["sides"]:
    if truth != 1:
        continue
    wn = -(wcl @ nn); fcl = ((wcl @ P) / wn, (wcl @ U) / wn)
    segs = fibre.trace_arc_rows(m, P, U, 0.0, math.pi, fibre.Ops())
    worst = []
    for t0, t1, xm in segs:
        if t1 - t0 < 1e-6:
            continue
        st0 = fibre.State(m, xm, P, U, nn, fibre.Ops())
        for th in np.linspace(t0, t1, 12)[1:-1]:
            fibre.LOOK = 2
            iv2, _ = fibre.trace_F2(m, st0, P, U, nn, truth, fcl, th, fibre.Ops(), {})
            ivx, _ = fibre.fibre_trace(m, st0, P, U, nn, truth, fcl, th, "FX", fibre.Ops(), {})
            d2, dx, db = dens(iv2, th), dens(ivx, th), brute(P, U, nn, truth, fcl, th)
            if abs(d2 - db) > 2e-4 or abs(dx - db) > 2e-4:
                worst.append((th, d2, dx, db, iv2, ivx, t0, t1))
    print(f"side truth={truth}: {len(worst)} theta samples with pointwise mismatch > 1e-4")
    for th, d2, dx, db, iv2, ivx, t0, t1 in worst[:40]:
        c, s = math.cos(th), math.sin(th)
        fmt = lambda iv: [(round(0 if lo is None else lo[0] * c + lo[1] * s, 4), round(0 if hi is None else hi[0] * c + hi[1] * s, 4)) for lo, hi in iv]
        print(f" seg ({t0:.4f},{t1:.4f}) th {th:.4f} F2L2 {d2:.5f} FX {dx:.5f} brute {db:.5f}  F2L2 u-intervals {fmt(iv2)}  FX {fmt(ivx)}  clamp u {fcl[0]*c+fcl[1]*s:.4f}")

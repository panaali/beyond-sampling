"""T97: find the theta-validity pieces where F2L2's closed-form integral disagrees with a quadrature of its own
pointwise density (i.e. combinatorics change inside a piece the constraints declared valid), on the adversarial
H=6 model, family circle (0,1,2), s=-1, truth 1. usage: python diag_neg3.py"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder"))
import fibre  # noqa: E402
from fiber3 import FAMS, _Sint  # noqa: E402
from diag_neg2 import dens, brute  # noqa: E402

J = json.load(open(os.path.join(HERE, "adv_H6.json")))
m = tuple(np.array(w) for w in J["worst"]["model"])
fibre.LOOK = 2
fam = [f for f in FAMS if f["circle"] == (0, 1, 2) and f["s"] == -1][0]
P, U = fam["P"], fam["U"]
gx, gw = np.polynomial.legendre.leggauss(16)
tot_cf = tot_q = tot_b = 0.0
for nn, truth, wcl in fam["sides"]:
    if truth != 1:
        continue
    wn = -(wcl @ nn); fcl = ((wcl @ P) / wn, (wcl @ U) / wn)
    for t0, t1, xm in fibre.trace_arc_rows(m, P, U, 0.0, math.pi, fibre.Ops()):
        st0 = fibre.State(m, xm, P, U, nn, fibre.Ops()); cache = {}
        stack = [(t0, t1)]
        while stack:
            a0, a1 = stack.pop()
            if a1 - a0 < 1e-13:
                continue
            tm = 0.5 * (a0 + a1)
            iv, cons = fibre.trace_F2(m, st0, P, U, nn, truth, fcl, tm, fibre.Ops(), cache)
            a, b = cons.interval(tm, a0, a1, fibre.Ops())
            cf = sum(_Sint(hi[0], hi[1], a, b) - (0 if lo is None else _Sint(lo[0], lo[1], a, b)) for lo, hi in iv)
            # quadrature of the same intervals' pointwise density, re-traced at each node
            ths = 0.5 * (a + b) + 0.5 * (b - a) * gx
            q = 0.5 * (b - a) * sum(w * dens(fibre.trace_F2(m, st0, P, U, nn, truth, fcl, th, fibre.Ops(), {})[0], th) for th, w in zip(ths, gw))
            qb = 0.5 * (b - a) * sum(w * brute(P, U, nn, truth, fcl, th, 20001) for th, w in zip(ths, gw)) if b - a > 1e-3 else q
            tot_cf += cf; tot_q += q; tot_b += qb
            if abs(cf - q) > 1e-5:
                c, s = math.cos(tm), math.sin(tm)
                print(f"piece ({a:.5f},{b:.5f}) from tm {tm:.5f} in ({a0:.4f},{a1:.4f}): closed form {cf:.5f} vs requadrature {q:.5f} (brute {qb:.5f}); "
                      f"intervals at tm {[(round(0 if lo is None else lo[0]*c+lo[1]*s,4), round(hi[0]*c+hi[1]*s,4)) for lo,hi in iv]}")
            if a > a0:
                stack.append((a0, a))
            if b < a1:
                stack.append((b, a1))
print(f"family side total x 1/4pi: closed form {tot_cf / (4 * math.pi):.6f}, requadrature {tot_q / (4 * math.pi):.6f}, brute {tot_b / (4 * math.pi):.6f}")

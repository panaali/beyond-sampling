"""T399: verify an exact (no unstable neuron) open leaf of bab_cpu.py by forward evaluation: replay the best-first
splitting of one bench node until the popped leaf has no unstable neuron, take the LP maximiser of the margin
(L_p3 - max_j L_j) and run the network on it in float64. If argmax = p3, x_p1 >= x_p2 >= x_p3 + DELTA*, x_p3 >= lows,
the node contains true event points (scale x so that the gap exceeds any t0).
usage: python verify_leaf.py BENCH.npz IDX [MAXLEAVES]"""
import sys

import numpy as np

import os

import relax2

relax2.DELTA = float(os.environ.get("DELTA", relax2.DELTA))   # g23 normalisation (T398b: 1/3 at t0 = 2.5)

d = np.load(sys.argv[1]); i = int(sys.argv[2]); MAXL = int(sys.argv[3]) if len(sys.argv) > 3 else 512
tri = tuple(int(v) for v in d["tri"]); k = int(d["k"]); G = d["G"][i][:int(d["m"][i])]; A = d["A"][i]
src = open(__file__.replace("verify_leaf.py", "bab_cpu.py")).read()
exec(src[src.index("class Leaf"):src.index("res = []")])


def argmax_point(L, fix):
    P, bounds = L.program(L.nb, False, fix)
    ub, bu, eq, be = P
    ub = [np.r_[r, 0.] for r in ub]; eq = [np.r_[r, 0.] for r in eq]
    for r in L.ev_ub:
        ub.append(np.r_[r, 1.]); bu.append(0.)
    c = np.zeros(L.nv + 1); c[-1] = -1
    return relax2.solve(c, (ub, bu, eq, be), bounds + [(None, None)]).x[:relax2.D]


closed, mg, L = evaluate(G, A, {}); opened = [(-mg, 0, {}, L)]; n = 1; cnt = 1
while opened and n < MAXL:
    opened.sort(key=lambda a: a[0]); negm, _, fix, L = opened.pop(0); sj = pick(L, fix)
    if sj is None:
        x = argmax_point(L, fix); h = np.zeros(16)
        for t in range(10):
            h = np.maximum(relax2.Wh @ h + relax2.Wi * x[t], 0)
        lg = relax2.Wo @ h; p1, p2, p3 = tri; lows = [j for j in range(10) if j not in tri]
        print("exact leaf after", n, "leaves; LP margin %.6f" % -negm, "x", np.round(x, 4))
        print("argmax", int(lg.argmax()), "p3", p3, "true margin %.6f" % (lg[p3] - np.delete(lg, p3).max()),
              "order ok", x[p1] >= x[p2] >= x[p3], "gap23 %.4f" % (x[p2] - x[p3]), "gap23/|x|inf %.4f" % ((x[p2] - x[p3]) / np.abs(x).max()), "p3>=lows", bool(np.all(x[p3] >= x[lows])),
              "G x >= 0", bool(np.all(G @ x >= -1e-9)))
        break
    n += 1
    for f in (1, 0):
        cl, m2, L2 = evaluate(G, A, {**fix, sj: f}); cnt += 1
        if not cl:
            opened.append((-m2, cnt, {**fix, sj: f}, L2))

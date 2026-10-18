"""T344 option (b): exact activation-region growth of the first k steps inside a triple's far-gap sub-cone
(README, option (b) pre-registration). Same LP cone splitting as tie_tail_labels/prefix_blind.py, in 10-D.
usage: python cone_count.py KMAX -> results_cone_count.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-9
D = 10
TRIPLES = ((0, 1, 2), (3, 6, 9), (9, 5, 1))


def models():
    sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
    import r0
    from alg_zoo import zoo_2nd_argmax
    f = lambda w: w.detach().numpy().astype(np.float64)
    m = zoo_2nd_argmax(16, 10, seed=0)
    return {"orig": r0.target(), "seed0": (f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight))}


def cone(tri):
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]; rows = []
    e = np.eye(D)
    rows += [e[p1] - e[p2], e[p2] - e[p3]] + [e[p3] - e[l] for l in lows]          # ordering
    g12 = e[p1] - e[p2]; g23 = e[p2] - e[p3]
    rows.append(g23 - 2 * g12)                                                    # g23 >= 2 g12
    rows += [g23 - 2 * (e[p3] - e[l]) for l in lows]                              # g23 >= 2 (x_p3 - x_l)
    return np.array(rows)


def range_over(G, c):
    nc = np.linalg.norm(c)
    if nc == 0:
        return 0.0, 0.0
    out = []
    for s in (1.0, -1.0):
        r = linprog(s * c / nc, A_ub=-G, b_ub=np.zeros(len(G)), bounds=[(-1, 1)] * D, method="highs")
        out.append(s * r.fun if r.status == 0 else np.nan)
    return out[0], out[1]


def count(model, tri, kmax, maxcells=200_000):
    Wi, Wh, _ = model; H = len(Wi); res = []
    cells = [(cone(tri), np.zeros((H, D)), b"")]
    for t in range(kmax):
        t0 = time.time(); new = []; ok = True
        for G, A, hist in cells:
            Z = Wh @ A; Z[:, t] += Wi
            pieces = [(G, [])]
            for j in range(H):
                nxt = []
                for Gp, g in pieces:
                    lo, hi = range_over(Gp, Z[j])
                    if hi <= TOL:
                        nxt.append((Gp, g + [0]))
                    elif lo >= -TOL:
                        nxt.append((Gp, g + [1]))
                    else:
                        nxt.append((np.vstack([Gp, Z[j]]), g + [1])); nxt.append((np.vstack([Gp, -Z[j]]), g + [0]))
                pieces = nxt
            for Gp, g in pieces:
                g = np.array(g, float); new.append((Gp, g[:, None] * Z, hist + np.packbits(g > 0).tobytes()))
            if len(new) > maxcells:
                ok = False; break
        cells = new
        res.append(dict(k=t + 1, complete=ok, cells=len(cells), regions=len({h for _, _, h in cells}), sec=round(time.time() - t0, 1)))
        print(tri, res[-1], flush=True)
        if not ok:
            break
    return res


if __name__ == "__main__":
    kmax = int(sys.argv[1]); out = {}
    for nm, m in models().items():
        for tri in TRIPLES:
            out[f"{nm}|{tri}"] = count(m, tri, kmax)
            json.dump(out, open(os.path.join(HERE, "results_cone_count.json"), "w"), indent=1)

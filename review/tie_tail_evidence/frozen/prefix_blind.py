"""T218 blind sub-test: exact gate-pattern regions of the target's k-step prefixes, restricted to inputs with at most
one positive coordinate among x_0..x_{k-1} (the blind region, FINDINGS sec. 3).

Independent of T192's prefix.py (which needs its geometry stack): every cell is a polyhedral cone {G x >= 0}; at each
step each neuron's pre-activation c.x is tested by two LPs over cone & [-1,1]^k and the cone is split when c.x takes
both signs (> tol and < -tol, so both pieces are full-dimensional). Leaves = cells; distinct gate histories = regions.
--full runs the same code on all of R^k (2^k orthant cones) to check against T192's counts (k2..k5: 34, 792, 21032,
632394 regions).

usage: python prefix_blind.py KMAX [--full] [--maxcells N]  -> results/prefix_blind[_full].json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-9


def target():
    from alg_zoo import example_2nd_argmax
    m = example_2nd_argmax().eval()
    f = lambda w: w.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)


def range_over(G, c, k):
    """(min, max) of c.x over {G x >= 0} & [-1,1]^k (scaled by |c|)."""
    nc = np.linalg.norm(c)
    if nc == 0:
        return 0.0, 0.0
    A = -G if len(G) else None; bnd = np.zeros(len(G)) if len(G) else None
    out = []
    for s in (1.0, -1.0):
        r = linprog(s * c / nc, A_ub=A, b_ub=bnd, bounds=[(-1, 1)] * k, method="highs")
        out.append(s * r.fun if r.status == 0 else np.nan)
    return out[0], out[1]


def start_cones(k, full):
    I = np.eye(k)
    if full:
        signs = ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1
        return [I * s[:, None] for s in signs]
    out = [-I]                                                     # all <= 0
    for i in range(k):
        S = -np.ones(k); S[i] = 1
        out.append(I * S[:, None])                                 # x_i >= 0, others <= 0
    return out


def run(kmax, full=False, maxcells=200_000):
    Wi, Wh, _ = target(); H = len(Wi)
    res = dict(kmax=kmax, full=full, steps=[])
    for k in range(1, kmax + 1):
        t0 = time.time(); nlp = 0
        cells = [(G, np.zeros((H, k)), b"") for G in start_cones(k, full)]
        ok = True
        for t in range(k):
            new = []
            for G, A, hist in cells:
                Z = Wh @ A; Z[:, t] += Wi                          # pre-activations z = Z x on this cell
                pieces = [(G, [])]
                for j in range(H):
                    nxt = []
                    for Gp, g in pieces:
                        lo, hi = range_over(Gp, Z[j], k); nlp += 2
                        if hi <= TOL:
                            nxt.append((Gp, g + [0]))
                        elif lo >= -TOL:
                            nxt.append((Gp, g + [1]))
                        else:
                            nxt.append((np.vstack([Gp, Z[j]]), g + [1]))
                            nxt.append((np.vstack([Gp, -Z[j]]), g + [0]))
                    pieces = nxt
                for Gp, g in pieces:
                    g = np.array(g, float)
                    new.append((Gp, g[:, None] * Z, hist + np.packbits(g > 0).tobytes()))
                if len(new) > maxcells:
                    ok = False; break
            cells = new
            if not ok:
                break
        r = dict(k=k, complete=ok, cells=len(cells), regions=len({h for _, _, h in cells}), lps=nlp,
                 sec=round(time.time() - t0, 1))
        res["steps"].append(r); print(json.dumps(r), flush=True)
        if not ok:
            break
    for a, b in zip(res["steps"][:-1], res["steps"][1:]):
        if b["complete"]:
            b["growth_regions"] = b["regions"] / a["regions"]; b["growth_cells"] = b["cells"] / a["cells"]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"prefix_blind{'_full' if full else ''}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    mc = int(a[a.index("--maxcells") + 1]) if "--maxcells" in a else 200_000
    run(int(a[0]), "--full" in a, mc)

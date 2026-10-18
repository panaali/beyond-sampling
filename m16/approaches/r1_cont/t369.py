"""T369 step 1: quadrature convergence of the graded g12 rule (README, T369 pre-registration).
usage: T340_SRC=... python t369.py RULE MODEL [MODEL ...] -> results_t369_<RULE>.json  (RULE in R1, R2, R3)"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t366 as X  # noqa: E402

RULES = {"R1": ([(0.0, 0.01), (0.01, 0.03), (0.03, 0.07), (0.07, 0.15), (0.15, 0.4)], 2),
         "R2": ([(0.0, 0.01), (0.01, 0.03), (0.03, 0.07), (0.07, 0.15), (0.15, 0.4)], 4),
         "R3": ([(a, b) for a, b in zip([0, .005, .01, .02, .03, .05, .07, .1, .15, .25], [.005, .01, .02, .03, .05, .07, .1, .15, .25, .4])], 2),
         "R4": ([(a, b) for a, b in zip([0, .005, .01, .02, .03, .05, .07, .1, .15, .25], [.005, .01, .02, .03, .05, .07, .1, .15, .25, .4])], 4)}
PAIRS = {"dev3_s547": (2.25, 2.5), "dev3_s541": (1.75, 2.0), "dev3_s546": (2.5,), "dev3_s543": (2.5,)}


CHUNK = 60_000


def est_chunked(model, t0s, tmax=None):
    """X.est_e1 with the rows walked in chunks (memory); estimates identical (rows are independent), FLOPs as executed."""
    tmax = X.rare.TBIG if tmax is None else tmax
    model = tuple(np.asarray(w, np.float64) for w in model); res = {t0: 0.0 for t0 in t0s}; flops = 0.0; ts = min(t0s)
    for ri, tab in enumerate(X.G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = X.LINES[ri]; nrow = len(x)
        k = np.tile(X.TRI[:, 1], nrow // X.NT); k3 = np.tile(X.TRI[:, 2], nrow // X.NT)
        node_of_row = np.repeat(np.arange(len(nodes)), X.NN * X.NT); pieces = []
        for c0 in range(0, nrow, CHUNK):
            sl = slice(c0, min(c0 + CHUNK, nrow))
            p, _, f = X.S.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], ts, tmax); flops += f
            pieces += [(r + c0, lo, hi) for r, lo, hi in p]
        wgt = tab["dens"] * tab["blind12"]
        for t0 in t0s:
            W, f = X.rare.mass(pieces, nrow, node_of_row, nodes, t0); flops += f
            res[t0] += float((W * om[:, None]).sum(0) * wgt @ wts) / X.NT; flops += nrow * len(nodes) * 3
    return res, flops


def set_rule(name):
    panels, npt = RULES[name]
    X.PANELS = panels; X._gx, X._gw = np.polynomial.legendre.leggauss(npt); X.NN = npt * len(panels) + X.K_TAIL
    X.LINES = X.node_lines_graded()


if __name__ == "__main__":
    rule, models = sys.argv[1], sys.argv[2:]; set_rule(rule)
    z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
    truth = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "t363", "t367_truth_gpu.json")))
    fn = os.path.join(HERE, f"results_t369_{rule}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in models:
        i = names.index(nm); m = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
        t0 = time.time(); r, f = est_chunked(m, PAIRS[nm])
        rec = dict(C=f / X.TC.cf.FPE, est={str(k): v for k, v in r.items()}, ratio={})
        for t in PAIRS[nm]:
            key = next((k for k in truth if k.startswith(nm) and abs(float(k.split("@")[1]) - t) < 1e-9), None)
            p = truth[key]["p"] if key and isinstance(truth[key], dict) and "p" in truth[key] else None
            rec["ratio"][str(t)] = r[t] / p if p else None
        out[nm] = rec; json.dump(out, open(fn, "w"), indent=1)
        print(rule, nm, {k: f"{v:.3e}" for k, v in rec["est"].items()}, "ratio", rec["ratio"], "C %.0f" % rec["C"], round(time.time() - t0), flush=True)

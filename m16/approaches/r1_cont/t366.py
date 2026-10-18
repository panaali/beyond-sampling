"""T366: T364's frozen estimator (fixed walk, T354 segment start, T_hi = 6, K3 = T340c) with the graded g12 rule
clustered at the triple tie (README, T366 pre-registration). E1(t0) only.
usage: T340_SRC=... python t366.py -> results_t366_dev.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t340c as TC  # noqa: E402
import t354 as S  # noqa: E402
import walkfix as WF  # noqa: E402

rare = TC.rare; G = TC.G; NT = TC.NT; TRI = TC.TRI
rare.walk3 = lambda model, x0, d, k, k3, tmax, max_cells=5000: WF.walk3_fixed(G, model, x0, d, k, k3, tmax, max_cells)[:5]
PANELS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.07), (0.07, 0.15), (0.15, 0.4)]
A_TAIL, K_TAIL = 0.4, 3
_gx, _gw = np.polynomial.legendre.leggauss(2)


def graded_g12(v2):
    """Nodes and weights (summing to 1) for the law of g12 | v2, density phi(v2 + g) / (1 - Phi(v2)) on g >= 0."""
    norm_ = ndtr(-v2); xs, ws = [], []
    for a, b in PANELS:
        g = a + (b - a) * (_gx + 1) / 2; xs += list(g); ws += list((b - a) / 2 * _gw * G._pdf(v2 + g) / norm_)
    gn, gw = TC.gauss_g12(v2, A_TAIL, K_TAIL); tail = ndtr(-(v2 + A_TAIL)) / norm_
    xs += list(gn); ws += list(gw * tail)
    return np.array(xs), np.array(ws)


NN = 2 * len(PANELS) + K_TAIL


def node_lines_graded():
    out = []
    for t in G.TABLES:
        xs, dxs, om = [], [], []
        for v2 in t["nodes"]:
            v3s = v2 - rare._cond_g23(v2); v4s = float(G.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
            m4 = float(G.mu(np.array([v4s]))[0]); dm4 = float(G.dmu(v4s)[0])
            gn, gw = graded_g12(v2)
            for g12, w in zip(gn, gw):
                x, dx = G.inputs(v2 + g12, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
                xs.append(x); dxs.append(dx); om.append(np.full(NT, w))
        out.append((np.concatenate(xs), np.concatenate(dxs), np.concatenate(om)))
    return out


LINES = node_lines_graded()                    # model-free, once


def est_e1(model, t0s, tmax=rare.TBIG):
    model = tuple(np.asarray(w, np.float64) for w in model); res = {t0: 0.0 for t0 in t0s}; flops = 0.0; t_start = min(t0s)
    for ri, tab in enumerate(G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = LINES[ri]; nrow = len(x)
        k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT); node_of_row = np.repeat(np.arange(len(nodes)), NN * NT)
        pieces, _, f = S.walk_seg(model, x, dx, k, k3, t_start, tmax); flops += f
        wgt = tab["dens"] * tab["blind12"]
        for t0 in t0s:
            W, f = rare.mass(pieces, nrow, node_of_row, nodes, t0); flops += f
            res[t0] += float((W * om[:, None]).sum(0) * wgt @ wts) / NT; flops += nrow * len(nodes) * 3
    return res, flops


def dev_models():
    z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
    out = {"orig": TC.cf.lb3_models()["orig"]}
    for nm in ("dev3_s526", "dev3_s527"):
        i = names.index(nm); out[nm] = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
    return out


def TRUTH(truth, nm, t):
    key = f"{nm}@{t:g}"
    return truth[key]["p"] if key in truth else None


if __name__ == "__main__":
    pairs = {"orig": (1.75, 2.0), "dev3_s526": (3.25, 3.5), "dev3_s527": (2.0, 2.25)}
    truth = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "t363", "t364_truth_gpu.json")))
    out = {}
    for nm, m in dev_models().items():
        t0 = time.time(); r, f = est_e1(m, pairs[nm]); rec = dict(C=f / TC.cf.FPE, est={str(k): v for k, v in r.items()})
        rec["truth"] = {}; rec["ratio"] = {}
        for t in pairs[nm]:
            p = TRUTH(truth, nm, t); rec["truth"][str(t)] = p; rec["ratio"][str(t)] = r[t] / p if p else None
        out[nm] = rec; json.dump(out, open(os.path.join(HERE, "results_t366_dev.json"), "w"), indent=1)
        print(nm, {k: f"{v:.3e}" for k, v in rec["est"].items()}, "ratio", {k: round(v, 2) for k, v in rec["ratio"].items()},
              "C %.0f" % rec["C"], round(time.time() - t0), flush=True)

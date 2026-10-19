"""T349 (T340c): T340's variant B with g12 integrated by a K-node Gauss rule for its exact law given v2, instead of being
pinned at E[g12 | v2] (README, T349 pre-registration). T340's code is used read-only from a pinned checkout
(claude/t286-cheap-frontier @620eb47b) given by the env var T340_SRC.
usage: python t340c.py gate        -> results_t340c_gate.json (B reproduced bitwise vs T340's results/est.json)
       python t340c.py run K       -> results_t340c_K{K}.json (est / est2 / C per model, est.json's shape)"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ["T340_SRC"]
sys.path.insert(0, os.path.join(SRC, "m16", "approaches", "rare_events"))
sys.path.insert(0, os.path.join(SRC, "m16", "approaches", "cheap_frontier"))
import cf  # noqa: E402
import rare  # noqa: E402

G = rare.G; NT = rare.NT; TRI = rare.TRI
NAMES = ("orig", "seed0", "seed3")


def gauss_g12(v2, a, K, n=40001):
    """K-node Gauss rule for g = v1 - v2 given v2, restricted to g >= a: density prop. to phi(v2 + g) (Stieltjes on a fine
    model-free grid, as r1w.gauss_v4). Returns nodes, weights (sum 1)."""
    g = np.linspace(a, a + 12.0, n); w = G._pdf(v2 + g); w = w / w.sum()
    al, be = np.zeros(K), np.zeros(K); p_prev = np.zeros_like(g); p = np.ones_like(g); nrm_prev = 1.0
    for j in range(K):
        nrm = (w * p * p).sum(); al[j] = (w * g * p * p).sum() / nrm
        if j > 0:
            be[j] = nrm / nrm_prev
        p_new = (g - al[j]) * p - (be[j] if j > 0 else 0) * p_prev
        p_prev, p, nrm_prev = p, p_new, nrm
    Jm = np.diag(al) + np.diag(np.sqrt(be[1:]), 1) + np.diag(np.sqrt(be[1:]), -1)
    x, U = np.linalg.eigh(Jm)
    return x, U[0] ** 2


def node_lines(K, a):
    """Per region: line23 inputs for every (v2 node j, g12 node k, triple i), row = (j K + k) NT + i; and row weights."""
    out = []
    for t in G.TABLES:
        xs, dxs, om = [], [], []
        for v2 in t["nodes"]:
            v3s = v2 - rare._cond_g23(v2); v4s = float(G.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
            m4 = float(G.mu(np.array([v4s]))[0]); dm4 = float(G.dmu(v4s)[0])
            gn, gw = gauss_g12(v2, a, K)
            for g12, w in zip(gn, gw):
                x, dx = G.inputs(v2 + g12, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
                xs.append(x); dxs.append(dx); om.append(np.full(NT, w))
        out.append((np.concatenate(xs), np.concatenate(dxs), np.concatenate(om)))
    return out


def est_c(model, K, t0s=(0.0, 1.0, 1.25, 1.5), t2s=(0.5, 0.75), tmax=rare.TBIG):
    model = tuple(np.asarray(w, np.float64) for w in model)
    res = {t0: 0.0 for t0 in t0s}; res2 = {t: 0.0 for t in t2s}; flops = 0.0
    fams = [("E1", 0.0)] + [("E2", t) for t in t2s]
    for fam, a in fams:
        lines = node_lines(K, a)
        for ri, tab in enumerate(G.TABLES):
            nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = lines[ri]; nrow = len(x)
            k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT)
            node_of_row = np.repeat(np.arange(len(nodes)), K * NT)
            pieces, pieces_c, f, it, ncell = rare.walk3(model, x, dx, k, k3, tmax); flops += f
            if fam == "E1":
                wgt = tab["dens"] * tab["blind12"]
                for t0 in t0s:
                    W, f = rare.mass(pieces, nrow, node_of_row, nodes, t0); flops += f
                    res[t0] += float((W * om[:, None]).sum(0) * wgt @ wts) / NT; flops += nrow * len(nodes) * 3
            else:
                Cm, f = rare.mass(pieces_c, nrow, node_of_row, nodes, a); flops += f
                tot = ndtr(nodes - a) ** 8 - ndtr(nodes - tmax) ** 8
                W = np.zeros_like(Cm); jj = node_of_row; W[np.arange(nrow), jj] = tot[jj] - Cm[np.arange(nrow), jj]
                w2 = tab["dens"] * ndtr(-(nodes + a))
                res2[a] += float(((W * om[:, None]) * w2[None]).sum(0) @ wts) / NT; flops += nrow * len(nodes) * 4
    return res, res2, flops


if __name__ == "__main__":
    ms = cf.lb3_models()
    if sys.argv[1] == "gate":
        ref = json.load(open(os.path.join(SRC, "m16", "approaches", "rare_events", "results", "est.json"))); out = {}
        for nm in NAMES:
            t0 = time.time(); r, r2, f, _ = rare.est(ms[nm], variant="B"); rf = ref[f"{nm}:B"]
            out[nm] = dict(E1_bitwise=all(r[float(k)] == v for k, v in rf["est"].items()),
                           E2_bitwise=all(r2[float(k)] == v for k, v in rf["est2"].items()), C_bitwise=bool(f / cf.FPE == rf["C"]),
                           sec=time.time() - t0)
            print(nm, out[nm], flush=True)
        json.dump(out, open(os.path.join(HERE, "results_t340c_gate.json"), "w"), indent=1)
    else:
        K = int(sys.argv[2]); out = {}
        for nm in NAMES:
            t0 = time.time(); r, r2, f = est_c(ms[nm], K)
            out[f"{nm}:C{K}"] = dict(est={str(k): v for k, v in r.items()}, est2={str(k): v for k, v in r2.items()}, C=f / cf.FPE,
                                     sec=time.time() - t0)
            print(nm, {k: f"{v:.3e}" for k, v in r.items()}, {k: f"{v:.3e}" for k, v in r2.items()}, f"C {f / cf.FPE:.0f}",
                  round(time.time() - t0), flush=True)
            json.dump(out, open(os.path.join(HERE, f"results_t340c_K{K}.json"), "w"), indent=1)

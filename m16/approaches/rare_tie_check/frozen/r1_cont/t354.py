"""T354: T340c (t340c.est_c) with each line walked only on the event's segment, from an exact restart state at the
segment start (README, T354 pre-registration). usage: T340_SRC=... python t354.py K -> results_t354_K{K}.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t340c as TC  # noqa: E402

rare = TC.rare; G = TC.G; NT = TC.NT; TRI = TC.TRI; cf = TC.cf


def walk_seg(model, x, dx, k, k3, t_start, tmax):
    """walk3 from t_start: fresh first cell at x + t_start dx (exact), pieces shifted back to absolute t."""
    p, pc, f, it, nc = rare.walk3(model, x + t_start * dx, dx, k, k3, tmax - t_start)
    sh = lambda P: [(r, lo + t_start, hi + t_start) for r, lo, hi in P]
    return sh(p), sh(pc), f


def est_seg(model, K, t0s=(1.0, 1.25, 1.5), t2s=(0.5, 0.75), tmax=rare.TBIG):
    model = tuple(np.asarray(w, np.float64) for w in model)
    res = {t0: 0.0 for t0 in t0s}; res2 = {t: 0.0 for t in t2s}; flops = 0.0
    for fam, a in [("E1", 0.0)] + [("E2", t) for t in t2s]:
        lines = TC.node_lines(K, a); t_start = min(t0s) if fam == "E1" else a
        for ri, tab in enumerate(G.TABLES):
            nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = lines[ri]; nrow = len(x)
            k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT)
            node_of_row = np.repeat(np.arange(len(nodes)), K * NT)
            pieces, pieces_c, f = walk_seg(model, x, dx, k, k3, t_start, tmax); flops += f
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
    K = int(sys.argv[1]); ms = cf.lb3_models(); out = {}
    ref = json.load(open(os.path.join(HERE, f"results_t340c_K{K}.json")))
    for nm in TC.NAMES:
        t0 = time.time(); r, r2, f = est_seg(ms[nm], K); rf = ref[f"{nm}:C{K}"]
        rel = {**{f"E1({k})": r[k] / rf["est"][str(k)] - 1 for k in r}, **{f"E2({k})": r2[k] / rf["est2"][str(k)] - 1 for k in r2}}
        out[f"{nm}:S{K}"] = dict(est={str(k): v for k, v in r.items()}, est2={str(k): v for k, v in r2.items()}, C=f / cf.FPE,
                                 C_ratio_vs_T349=f / cf.FPE / rf["C"], rel_vs_T349=rel, sec=time.time() - t0)
        print(nm, "C %.0f (x%.3f of T349 K%d)" % (f / cf.FPE, f / cf.FPE / rf["C"], K), "max rel dev %.2e" % max(abs(v) for v in rel.values()),
              round(time.time() - t0), flush=True)
        json.dump(out, open(os.path.join(HERE, f"results_t354_K{K}.json"), "w"), indent=1)

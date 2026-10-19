"""T362 conditional Monte Carlo along exact v3 lines. SAMPLING-CLASS (random model inputs), unbiased.
For x ~ N(0, I): v3 given everything else has law phi on (v4, v2) at position p3, so E[f] = E[Z] with
Z = int_{v4}^{v2} f(x | x_p3 = v) phi(v) dv / (Phi(v2) - Phi(v4)), computed exactly by rare.walk3 along x_p3.
usage: python cmc.py [lines per event] -> results/cmc.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402
import rare  # noqa: E402

EPS = 1e-9
TRUTH = {"E1(1.0)": 53668 / 1e10, "E2(0.5)": 937806 / 1e10}       # T192 GPU, orig


def run(model, event, nlines, seed, batch=200_000):
    rng = np.random.default_rng(seed); Zs = []; ndrawn = 0; flops = 0.0; cells = 0
    need = nlines
    while need > 0:
        x = rng.standard_normal((batch, 10)); o = np.argsort(-x, 1); r = np.arange(batch)
        v = x[r[:, None], o]; v1, v2, v4 = v[:, 0], v[:, 1], v[:, 3]
        if event == "E1(1.0)":
            t0 = 1.0; ok = (v2 - v4) > t0
        else:
            t0 = 0.5; ok = ((v2 - v4) > t0) & ((v1 - v2) >= t0)
        idx = np.flatnonzero(ok)
        if len(idx) >= need:                      # stop exactly at the need-th passing input
            cut = idx[need - 1] + 1; idx = idx[:need]
        else:
            cut = batch
        ndrawn += cut; need -= len(idx)
        Zs.append(np.zeros(cut - len(idx)))      # screened inputs: Z = 0 exactly (model-free)
        if not len(idx):
            continue
        xs = x[idx]; p2, p3 = o[idx, 1], o[idx, 2]; s = v2[idx] - EPS; lo4 = v4[idx]
        x0 = xs.copy(); x0[np.arange(len(idx)), p3] = s
        d = np.zeros_like(x0); d[np.arange(len(idx)), p3] = -1.0
        tmax = s - lo4
        pk3, pkc, f, it, nc = rare.walk3(model, x0, d, p2, p3, tmax); flops += f; cells += nc
        n = len(idx); mass = np.zeros(n)
        if event == "E1(1.0)":
            for rows, a, b in pk3:
                a = np.maximum(a, t0 - EPS); m = b > a
                np.add.at(mass, rows[m], ndtr(s[rows[m]] - a[m]) - ndtr(s[rows[m]] - b[m])); flops += m.sum() * (2 * rare.G.F_PHI + 2)
        else:
            cm = np.zeros(n)
            for rows, a, b in pkc:
                a = np.maximum(a, t0 - EPS); m = b > a
                np.add.at(cm, rows[m], ndtr(s[rows[m]] - a[m]) - ndtr(s[rows[m]] - b[m])); flops += m.sum() * (2 * rare.G.F_PHI + 2)
            win = ndtr(s - (t0 - EPS)) - ndtr(lo4)                      # mass of v in (v4, v2 - t0)
            mass = win - cm; flops += n * (2 * rare.G.F_PHI + 2)
        Z = mass / (ndtr(v2[idx]) - ndtr(lo4)); flops += n * 4
        Zs.append(Z)
    Z = np.concatenate(Zs)
    return Z, ndrawn, flops, cells


def main(nlines=10_000):
    m = tuple(np.asarray(w, float) for w in cf.lb3_models()["orig"]); out = {}
    for ev, seed in (("E1(1.0)", 362000), ("E2(0.5)", 362001)):
        t = time.time(); Z, nd, fl, cells = run(m, ev, nlines, seed)
        p = TRUTH[ev]; est = Z.mean(); se = Z.std(ddof=1) / np.sqrt(len(Z)); C = fl / cf.FPE
        var_mc = p * (1 - p) / C; G = var_mc / se ** 2
        nz = Z[Z > 0]
        out[ev] = dict(truth=p, est=float(est), se=float(se), z=float((est - p) / se), n_drawn=int(nd), n_lines=nlines, C_fpe=C,
                       fpe_per_line=C / nlines, cells_per_line=cells / nlines, gain=G,
                       frac_lines_nonzero=len(nz) / nlines, n_nonzero=int(len(nz)), mean_Z_given_nonzero=float(nz.mean()) if len(nz) else 0.0,
                       sec=time.time() - t)
        print(ev, {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in out[ev].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "cmc.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)

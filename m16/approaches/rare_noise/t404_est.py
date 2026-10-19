"""T404 estimates (DEDUCTIVE; frozen code read-only, hashes asserted as in est_t388.py).
  a  : graded-g12 (T366, t366.est_e1's loop via t388_chunked) at T_hi = 3.5 on E1(1.0, 1.25, 1.5)
  k3e2: K3 (T340c lines + walk fix + T354 segment start) E2 branch of t354.est_seg, T_hi = 6, row-chunked
  ae2: T340 variant A (rare.est(variant="A"), walk fix), E2 only
E2(t) = {network wrong, g12 >= t, g23 >= t}, t in T2S.
usage: python t404_est.py PART SHARD NSHARD -> results/t404_PART_SHARD.json;  python t404_est.py check_k3e2 MODEL"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import est_t388  # noqa: E402,F401  (asserts all 7 sha256 of t366_freeze_sha256.txt; t366 patches rare.walk3 = walkfix)
import models as MP  # noqa: E402
import t366 as T366  # noqa: E402
import t388_chunked as CK  # noqa: E402

TC, SG, rare, G = T366.TC, T366.S, T366.rare, T366.G
NT, TRI = TC.NT, TC.TRI
T0S = (1.0, 1.25, 1.5); T2S = (0.5, 0.75, 1.0); THI_A = 3.5
CH = 40_000; _LINES = {}


def k3_e2(model, t2s=T2S, K=3, tmax=rare.TBIG):
    """t354.est_seg's E2 branch (fam E2, a = t), lines walked in row chunks; equals est_seg's est2 up to summation order."""
    model = tuple(np.asarray(w, np.float64) for w in model); res2 = {t: 0.0 for t in t2s}; flops = 0.0
    for a in t2s:
        lines = _LINES.setdefault((K, a), TC.node_lines(K, a))
        for ri, tab in enumerate(G.TABLES):
            nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = lines[ri]; nrow = len(x)
            k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT); node_of_row = np.repeat(np.arange(len(nodes)), K * NT)
            tot = ndtr(nodes - a) ** 8 - ndtr(nodes - tmax) ** 8; w2 = tab["dens"] * ndtr(-(nodes + a)); acc = np.zeros(len(nodes))
            for s in range(0, nrow, CH):
                sl = slice(s, min(s + CH, nrow)); n = sl.stop - sl.start; jj = node_of_row[sl]
                _, pieces_c, f = SG.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], a, tmax); flops += f
                Cm, f = rare.mass(pieces_c, n, jj, nodes, a); flops += f
                W = np.zeros_like(Cm); W[np.arange(n), jj] = tot[jj] - Cm[np.arange(n), jj]
                acc += (W * om[sl][:, None]).sum(0)
            res2[a] += float((acc * w2) @ wts) / NT; flops += nrow * len(nodes) * 4
    return res2, flops


def a_e2(model, t2s=T2S):
    _, r2, f, _ = rare.est(tuple(np.asarray(w, np.float64) for w in model), t0s=(), variant="A", t2s=t2s)
    return r2, f


RUN = {"a": lambda m: CK.est_e1_chunked(T366, m, T0S, tmax=THI_A), "k3e2": k3_e2, "ae2": a_e2}

if __name__ == "__main__":
    P = MP.panel()
    if sys.argv[1] == "check_k3e2":            # vs the unchunked frozen est_seg on one model (E1 at t0 = 1.0 also walked)
        nm = sys.argv[2]; r, f = k3_e2(P[nm]); _, r2, f2 = SG.est_seg(P[nm], 3, t0s=(1.0,), t2s=T2S)
        out = dict(model=nm, rel={str(t): r[t] / r2[t] - 1 for t in T2S}, est={str(t): r[t] for t in T2S})
        print(out); json.dump(out, open(os.path.join(HERE, "results", "t404_check_k3e2.json"), "w"), indent=1); sys.exit()
    part, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); names = sorted(P)[shard::nsh]
    fn = os.path.join(HERE, "results", f"t404_{part}_{shard}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in names:
        if nm in out:
            continue
        t = time.time(); r, f = RUN[part](P[nm])
        out[nm] = dict(est={str(k): float(v) for k, v in r.items()}, C_fpe=f / 5948.0, sec=time.time() - t)
        json.dump(out, open(fn, "w"), indent=1)
        print(nm, {k: f"{v:.3e}" for k, v in out[nm]["est"].items()}, "C %.0f" % out[nm]["C_fpe"], "%.0fs" % out[nm]["sec"], flush=True)

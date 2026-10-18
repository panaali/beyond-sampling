"""T304 estimator R1W (deterministic; README for step labels): T302's exact segment walks with closure c (top-4, v4 at
the triple's latest free position, its value by a K-node Gauss rule for its exact conditional law given v3; 6 lows at
mu(v4)). est(model, budget=None, K=1, blind12=False, tmax=.5) -> (acc, charged FLOPs)."""
import os
import sys

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r0_walk")); sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
import r0  # noqa: E402
import walk as W  # noqa: E402

DEDUCTIVE = True
TRI = r0.TRIPLES; NT = len(TRI)
FREE = np.array([[p for p in range(10) if p not in t] for t in TRI])          # (720, 7) free positions, ascending
P4 = FREE[:, -1]                                                               # declared v4 position: the latest
LOWS = FREE[:, :-1]                                                            # the other 6


def gauss_v4(v3, K):
    """K-node Gauss rule for the law of v4 = max of 7 iid N(0,1) | < v3 (Stieltjes on a fine model-free grid)."""
    v = np.linspace(v3 - 8, v3, 40001)[:-1]
    w = norm.pdf(v) * ndtr(v) ** 6; w /= w.sum()
    a, b = np.zeros(K), np.zeros(K); p_prev = np.zeros_like(v); p = np.ones_like(v)
    for j in range(K):
        nrm = (w * p * p).sum(); a[j] = (w * v * p * p).sum() / nrm
        if j > 0:
            b[j] = nrm / nrm_prev
        p_new = (v - a[j]) * p - (b[j] if j > 0 else 0) * p_prev
        p_prev, p, nrm_prev = p, p_new, nrm
    J = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)
    x, U = np.linalg.eigh(J)
    return x, U[0] ** 2


def dmu(v):
    m = r0.mu(np.atleast_1d(v))
    return -np.atleast_1d(v) * m + m * m


def inputs(v1, v2, v3, v4, low, d1, d3, d4, dlow):
    n = NT; r = np.arange(n)
    x = np.zeros((n, 10)); dx = np.zeros((n, 10))
    for cols, val, dv in ((TRI[:, 0], v1, d1), (TRI[:, 1], v2, 0.0), (TRI[:, 2], v3, d3), (P4, v4, d4)):
        x[r, cols] = val; dx[r, cols] = dv
    for j in range(6):
        x[r, LOWS[:, j]] = low; dx[r, LOWS[:, j]] = dlow
    return x, dx


def est(model, budget=None, K=1, blind12=False, tmax=0.5, parts=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = TRI[:, 1]; fpe = 0.0; nphi = 0; tot = np.zeros(NT); tail = (np.arange(NT), np.full(NT, tmax), np.full(NT, W.INF))
    full = [(np.arange(NT), np.zeros(NT), np.full(NT, W.INF))]
    for reg, nodes, wts in (("nb", W.V2NB, W.W_NB), ("b", W.V2B, W.W_B)):
        v2s, g12s, g23s = W.TP[reg]; v3s = v2s - g23s
        xs, ws = gauss_v4(v3s, K)
        for v4s, w4 in zip(xs, ws):
            delta = v3s - v4s; m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(dmu(v4s)[0])
            # g23 line: v3 = v2* - t, v4 = v3 - delta, lows = mu(v4*) + mu'(v4*) (v4 - v4*)   (declared)
            x, dx = inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)
            p23, c23 = W.walk_line(model, x, dx, k, tmax); fpe += c23.sum()
            I23, a = W._profile(p23 + [tail], NT, nodes, "23"); nphi += a
            if reg == "nb" or blind12:
                x, dx = inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0)          # g12 line, exact closure
                p12, c12 = W.walk_line(model, x, dx, k, tmax); fpe += c12.sum()
                I12, b = W._profile(p12 + [tail], NT, nodes, "12"); nphi += b
            else:
                I12, b = W._profile(full, NT, nodes, "12"); nphi += b               # blind: g12 assumed correct
            tot += w4 * ((I12 * I23 * (90 * norm.pdf(nodes))[None]) @ wts); nphi += NT * len(nodes)
    flops = fpe * r0.FPE + nphi * W.FLOP_PHI + 4 * NT * W.G * 10 * K
    acc = float(tot.mean())
    return (acc, flops, dict(fpe_walk=float(fpe))) if parts else (acc, flops)


if __name__ == "__main__":
    for K in (1, 2, 3):
        x, w = gauss_v4(0.665, K); print("K", K, "nodes", x.round(4), "weights", w.round(4), "sum", w.sum().round(12))

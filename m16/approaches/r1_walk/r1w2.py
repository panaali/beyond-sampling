"""T306 estimator family (R1W refinements A / B / A+B; see README T306). est(model, budget=None, A=False, B=False)
-> (acc, charged FLOPs). V1 = est(A=False, B=False) reproduces r1w.est(K=1)."""
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r0_walk")); sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
import r0  # noqa: E402
import walk as W  # noqa: E402
import r1w  # noqa: E402

DEDUCTIVE = True
TMAX = 0.5
A_TMAX, A_CELLS = 0.10, 4
NT = r1w.NT


def _regions(B):
    """(name, v2 lo, v2 hi, transverse point (v2, g12, g23)) from a seeded model-free order-statistic table."""
    x = -np.sort(-np.random.default_rng(0).standard_normal((1_000_000, 10)), 1)[:, :3]
    v2 = x[:, 1]
    cuts = [("nb", 0.0, 6.5), ("b", -6.5, 0.0)]
    if B:
        m_nb = float(np.median(v2[v2 > 0])); m_b = float(np.median(v2[v2 <= 0]))
        cuts = [("nb1", 0.0, m_nb), ("nb2", m_nb, 6.5), ("b1", -6.5, m_b), ("b2", m_b, 0.0)]
    out = []
    for name, lo, hi in cuts:
        s = (v2 > lo) & (v2 <= hi); y = x[s]
        out.append((name, lo, hi, (y[:, 1].mean(), (y[:, 0] - y[:, 1]).mean(), (y[:, 1] - y[:, 2]).mean())))
    return out


REG = {False: _regions(False), True: _regions(True)}


def est(model, budget=None, A=False, B=False, parts=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    TRI = r1w.TRI; k = TRI[:, 1]; fpe = 0.0; nphi = 0; tot = np.zeros(NT)
    full = [(np.arange(NT), np.zeros(NT), np.full(NT, W.INF))]
    for name, lo, hi, (v2s, g12s, g23s) in REG[B]:
        nodes, wts = W._gl(lo, hi)
        v3s = v2s - g23s
        xs, ws = r1w.gauss_v4(v3s, 1); v4s = float(xs[0])
        delta = v3s - v4s; m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
        tail = (np.arange(NT), np.full(NT, TMAX), np.full(NT, W.INF))
        x, dx = r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)
        p23, c23 = W.walk_line(model, x, dx, k, TMAX); fpe += c23.sum()
        I23, a = W._profile(p23 + [tail], NT, nodes, "23"); nphi += a
        x, dx = r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0)
        if name.startswith("nb"):
            p12, c12 = W.walk_line(model, x, dx, k, TMAX); fpe += c12.sum()
            I12, b = W._profile(p12 + [tail], NT, nodes, "12")
        elif A:
            p12, c12, reach = walk_reach_t(model, x, dx, k, A_TMAX, A_CELLS); fpe += c12.sum()
            tl = (np.arange(NT), reach, np.full(NT, W.INF))           # beyond where the walk stopped: assumed correct
            I12, b = W._profile(p12 + [tl], NT, nodes, "12")
        else:
            I12, b = W._profile(full, NT, nodes, "12")
        nphi += b
        tot += (I12 * I23 * (90 * norm.pdf(nodes))[None]) @ wts; nphi += NT * len(nodes)
    flops = fpe * r0.FPE + nphi * W.FLOP_PHI + 4 * NT * W.G * 10 * len(REG[B])
    acc = float(tot.mean())
    return (acc, flops, dict(fpe_walk=float(fpe))) if parts else (acc, flops)


def walk_reach_t(model, x0, d, k, tmax, max_cells, eps=1e-10):
    """Copy of W.walk_line that also returns the parameter each row reached (cell cap)."""
    Wi, Wh, Wo = model; n = len(x0); T = x0.shape[1]
    t = np.zeros(n); restart = np.zeros(n, int); cost = np.zeros(n); first = np.ones(n, bool)
    pieces = []; act = np.arange(n); it = 0
    while len(act) and it < max_cells:
        it += 1
        X0, D = x0[act], d[act]; tc = t[act] + eps * (1 + np.abs(t[act]))
        h = np.zeros((len(act), Wh.shape[0])); dh = np.zeros_like(h)
        nxt = np.full(len(act), tmax); rstep = np.full(len(act), T)
        for s in range(T):
            z = h @ Wh.T + (X0[:, s] + tc * D[:, s])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, s:s + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc[:, None] - z / dz
            r = np.where((r > tc[:, None]) & np.isfinite(r), r, np.inf).min(1)
            upd = r < nxt; nxt = np.where(upd, r, nxt); rstep = np.where(upd, s, rstep)
            g = z > 0; h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        L = h @ Wo.T; dL = dh @ Wo.T
        lo, hi = W.correct_interval(L, dL, tc, k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        cost[act] += np.where(first[act], 2.0, 2.0 * (T - restart[act]) / T)
        first[act] = False; t[act] = nxt; restart[act] = rstep
        act = act[nxt < tmax]
    return pieces, cost, np.minimum(t, tmax)

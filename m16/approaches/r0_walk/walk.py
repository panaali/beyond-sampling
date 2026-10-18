"""T302 estimator WALK (deterministic; see README for step labels). est(model, budget=None) -> (acc, charged FLOPs).
Exact segment walks of R0's piecewise-linear margin along the g23 and g12 lines at declared transverse points, per
position triple, in the nonblind (v2 > 0) and blind (v2 <= 0) regions; exact model-free masses of the product of the
per-line correct sets."""
import os
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

DEDUCTIVE = True           # every step deductive or a declared structural approximation (README)
TMAX = 6.0
INF = 1e9           # marker for an interval extending to infinity
G = 48
FLOP_PHI = 20
_t, _w = np.polynomial.legendre.leggauss(G)


def _gl(a, b):
    return a + (b - a) * (_t + 1) / 2, (b - a) / 2 * _w


V2NB, W_NB = _gl(0.0, 6.5)
V2B, W_B = _gl(-6.5, 0.0)


def _transverse():
    """Conditional means of (v2, g12, g23) given v2 > 0 / v2 <= 0 (model-free order-statistic table, seeded)."""
    x = -np.sort(-np.random.default_rng(0).standard_normal((1_000_000, 10)), 1)[:, :3]
    out = {}
    for key, m in (("nb", x[:, 1] > 0), ("b", x[:, 1] <= 0)):
        y = x[m]; out[key] = (y[:, 1].mean(), (y[:, 0] - y[:, 1]).mean(), (y[:, 1] - y[:, 2]).mean())
    return out


TP = _transverse()


def correct_interval(L, dL, tc, k, lo, hi):
    """Correct set of target k within [lo, hi] for logits L + (t - tc) dL (an interval; numpy first-index ties)."""
    rr = np.arange(len(L))
    c = (L[rr, k][:, None] - L) - (dL[rr, k][:, None] - dL) * tc[:, None]
    e = dL[rr, k][:, None] - dL
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -c / e
    lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
    jj = np.arange(L.shape[1])[None]
    bad0 = (e == 0) & ((c < 0) | ((np.abs(c) <= 1e-12 * (1 + np.abs(L).max(1))[:, None]) & (jj < k[:, None])))
    lb[rr, k] = -np.inf; ub[rr, k] = np.inf
    clo = np.maximum(lb.max(1), lo); chi = np.minimum(ub.min(1), hi)
    ok = (chi > clo) & ~bad0.any(1)
    return np.where(ok, clo, 0.0), np.where(ok, chi, 0.0)


def walk_line(model, x0, d, k, tmax=TMAX, eps=1e-10, max_cells=400):
    """Exact walk of x0 + t d, t in [0, tmax]. Returns the correct intervals (list of (rows, lo, hi)) and cost (FPE)."""
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
        lo, hi = correct_interval(L, dL, tc, k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        cost[act] += np.where(first[act], 2.0, 2.0 * (T - restart[act]) / T)
        first[act] = False; t[act] = nxt; restart[act] = rstep
        act = act[nxt < tmax]
    return pieces, cost


def _line_inputs(tri, v1, v2, v3, lows, d1, d3, dlow):
    n = len(tri); r = np.arange(n)
    x = np.repeat(np.broadcast_to(lows, (n,))[:, None], r0.T, 1)
    dx = np.repeat(np.broadcast_to(np.asarray(dlow, float), (n,))[:, None], r0.T, 1)
    for col, val, dv in ((0, v1, d1), (1, v2, 0.0), (2, v3, d3)):
        x[r, tri[:, col]] = val; dx[r, tri[:, col]] = dv
    return x, dx


def _profile(pieces, n, nodes, kind, tmax=TMAX):
    """Per row and GL node: kind '12' -> sum over intervals of Phi(v2+b)-Phi(v2+a); '23' -> sum of
    Phi(v2-a)^8 - Phi(v2-b)^8 (an interval reaching TMAX is extended to infinity)."""
    out = np.zeros((n, len(nodes))); nphi = 0
    for rows, lo, hi in pieces:
        m = hi > lo
        if not m.any():
            continue
        rr, a, b = rows[m], lo[m], hi[m]
        binf = b >= INF - 1e-12
        if kind == "12":
            val = np.where(binf[:, None], 1.0, ndtr(nodes[None] + b[:, None])) - ndtr(nodes[None] + a[:, None])
        else:
            val = ndtr(nodes[None] - a[:, None]) ** 8 - np.where(binf[:, None], 0.0, ndtr(nodes[None] - b[:, None]) ** 8)
        np.add.at(out, rr, val); nphi += 2 * m.sum() * len(nodes)
    return out, nphi


def est(model, budget=None, parts=False, tmax=TMAX, blind_walk=True):
    model = tuple(np.asarray(w, np.float64) for w in model)
    tri = r0.TRIPLES; n = len(tri); k = tri[:, 1]; fpe = 0.0; nphi = 0; res = {}
    for reg, nodes, w in (("nb", V2NB, W_NB), ("b", V2B, W_B)):
        v2s, g12s, g23s = TP[reg]; v3s = v2s - g23s
        mu0 = float(r0.mu(np.array([v3s]))[0]); m = float(-v3s * mu0 + mu0 * mu0)   # mu'(v3*) = -v mu + mu^2
        one = np.ones(n)
        # g23 line: v3 = v2* - t, lows = mu(v3*) + mu'(v3*) (v3 - v3*) (linearised closure)
        x, dx = _line_inputs(tri, (v2s + g12s) * one, v2s * one, v2s * one, (mu0 + m * (v2s - v3s)) * one,
                             0.0, -1.0, -m)
        p23, c23 = walk_line(model, x, dx, k, tmax)
        # g12 line: v1 = v2* + t at (v2*, g23*); lows = mu(v3*) (exact R0)
        x, dx = _line_inputs(tri, v2s * one, v2s * one, v3s * one, mu0 * one, 1.0, 0.0, 0.0)
        p12, c12 = walk_line(model, x, dx, k, tmax)
        fpe += c23.sum() + c12.sum()
        tail = (np.arange(n), np.full(n, tmax), np.full(n, INF))     # beyond the walked range: assumed correct
        p23 = p23 + [tail]; p12 = p12 + [tail]
        I23, a = _profile(p23, n, nodes, "23"); I12, b = _profile(p12, n, nodes, "12"); nphi += a + b
        from scipy.stats import norm
        res[reg] = (I12 * I23 * (90 * norm.pdf(nodes))[None]) @ w            # correct mass per triple
        nphi += n * len(nodes)
    acc = float((res["nb"] + res["b"]).mean())
    flops = fpe * r0.FPE + nphi * FLOP_PHI + 4 * n * G * 10
    if parts:
        return acc, flops, dict(nb=float(res["nb"].mean()), b=float(res["b"].mean()), fpe_walk=float(fpe))
    return acc, flops


if __name__ == "__main__":
    # model-free check through the estimator's own code path: S12 = S23 = [0, inf) must give total mass 1
    from scipy.stats import norm
    rows = np.arange(3); full = [(rows, np.zeros(3), np.full(3, INF))]
    tot = 0.0
    for nodes, w in ((V2NB, W_NB), (V2B, W_B)):
        I12, _ = _profile(full, 3, nodes, "12"); I23, _ = _profile(full, 3, nodes, "23")
        tot += ((I12 * I23 * (90 * norm.pdf(nodes))[None]) @ w)[0]
    print("mass check:", tot, "transverse", {k: tuple(round(float(x), 3) for x in v) for k, v in TP.items()})

"""T402 design tools: line-CMC estimates (sampling-class, unbiased) of
  e1(model, t0, n, seed)  -> P(E1(t0)) and dP/dW (exact v3 lines under the conditional law given g23 > t0; = ../cmcgrad.py)
  acc(model, n, seed)     -> accuracy and d acc/dW (lines along a uniformly random coordinate j over R; target = position
                             of the 2nd largest, constant on the 3 segments cut by the 2nd and 1st largest of the others)
Both return dict(P/acc, se, grad, grad_se, C_fpe). grad=False skips the backward passes."""
import os
import sys

import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lgrad  # noqa: E402
import truthc_copy as TCC  # noqa: E402
import walk  # noqa: E402

PHI = lambda v: np.exp(-v ** 2 / 2) / np.sqrt(2 * np.pi)


def _draw_e1(c, t0, rng, n):
    v2 = np.interp(rng.random(n), c, TCC.GRID); v1 = -ndtri(rng.random(n) * ndtr(-v2))
    v3 = ndtri(ndtr(v2 - t0) * rng.random(n) ** (1 / 8)); lows = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))
    vals = np.column_stack([v1, v2, v3, lows]); perm = np.argsort(rng.random((n, 10)), 1)
    x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
    return x, perm, v2, lows.max(1)


def _accumulate(model, x0, d, k, lo, hi, nrm, wrow, nlines, grad):
    """Line masses of {argmax = k} on rows (several rows may share a line: wrow = row -> line), and boundary grads."""
    R, A, B, fl = walk.argmax_pieces(model, x0, d, k, lo, hi); R, A, B = walk.merge(R, A, B)
    Z = np.zeros(nlines); np.add.at(Z, wrow[R], (ndtr(B) - ndtr(A)) / nrm[R]); P_ = sum(int(np.prod(np.shape(w))) for w in model)
    G = np.zeros((nlines, P_)) if grad else None
    if grad:
        for e, keep in ((A, A > lo[R] + 1e-9 * (1 + np.abs(lo[R]))), (B, B < hi[R] - 1e-9 * (1 + np.abs(hi[R])))):
            rr, ee = R[keep], e[keep]
            if len(rr):
                Gm, _, _, f = lgrad.endpoint_grads(model, x0, d, k, rr, ee); fl += f
                np.add.at(G, wrow[rr], (PHI(ee) / nrm[rr])[:, None] * Gm)
    return Z, G, fl


def _finish(sZ, sZ2, sg, sg2, n, scale, fl):
    se = lambda s, s2: np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 0) / n)
    out = dict(P=scale * sZ / n, se=scale * se(sZ, sZ2), n=n, C_fpe=fl / 5948.0)
    if sg is not None:
        out.update(grad=scale * sg / n, grad_se=scale * se(sg, sg2))
    return out


def e1(model, t0, n, seed, grad=True, chunk=20_000):
    f = TCC.dens(t0); c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(TCC.GRID))]); c /= c[-1]
    pg = TCC.p_gap(t0); rng = np.random.default_rng(seed); sZ = sZ2 = 0.0; sg = sg2 = None; fl = 0.0; m0 = 0
    while m0 < n:
        m = min(chunk, n - m0); x, perm, v2, v4 = _draw_e1(c, t0, rng, m); p3 = perm[:, 2]; r = np.arange(m)
        d = np.zeros_like(x); d[r, p3] = 1.0; x0 = x.copy(); x0[r, p3] = 0.0; lo, hi = v4, v2 - t0
        Z, G, f = _accumulate(model, x0, d, p3, lo, hi, ndtr(hi) - ndtr(lo), r, m, grad); fl += f
        sZ += Z.sum(); sZ2 += (Z ** 2).sum()
        if grad:
            sg = G.sum(0) if sg is None else sg + G.sum(0); sg2 = (G ** 2).sum(0) if sg2 is None else sg2 + (G ** 2).sum(0)
        m0 += m
    return _finish(sZ, sZ2, sg, sg2, n, pg, fl)


def acc(model, n, seed, grad=True, chunk=20_000, L=9.0):
    rng = np.random.default_rng(seed); sZ = sZ2 = 0.0; sg = sg2 = None; fl = 0.0; m0 = 0
    while m0 < n:
        m = min(chunk, n - m0); x = rng.standard_normal((m, 10)); j = rng.integers(0, 10, m); r = np.arange(m)
        xo = x.copy(); xo[r, j] = -np.inf; o = np.argsort(-xo, 1); p1, p2 = o[:, 0], o[:, 1]; o1, o2 = xo[r, p1], xo[r, p2]
        x0 = x.copy(); x0[r, j] = 0.0; d = np.zeros_like(x); d[r, j] = 1.0
        X0 = np.repeat(x0, 3, 0); D = np.repeat(d, 3, 0); wrow = np.repeat(r, 3)
        K = np.column_stack([p2, j, p1]).ravel(); LO = np.column_stack([np.full(m, -L), o2, o1]).ravel()
        HI = np.column_stack([o2, o1, np.full(m, L)]).ravel(); nrm = np.ones(3 * m)
        Z, G, f = _accumulate(model, X0, D, K, LO, HI, nrm, wrow, m, grad); fl += f
        sZ += Z.sum(); sZ2 += (Z ** 2).sum()
        if grad:
            sg = G.sum(0) if sg is None else sg + G.sum(0); sg2 = (G ** 2).sum(0) if sg2 is None else sg2 + (G ** 2).sum(0)
        m0 += m
    return _finish(sZ, sZ2, sg, sg2, n, 1.0, fl)

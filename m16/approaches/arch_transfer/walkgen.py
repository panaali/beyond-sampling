"""Generic segment walks for any T432 candidate; drop-in for frozen/walk.py's `walk` (same arguments and outputs).

Along x(t) = y + t d, d = (e_a - e_b)/sqrt2, the misclassified 1-D mass int_{-tau}^{tau} 1[wrong at x(t)] dCDF(t) is
computed per segment. Target: R23 -> a for t > 0 and b for t < 0; R12 -> b for t > 0 and a for t < 0 (as walk.py).
- exact (net.mode = D, an int): the network is a fixed polynomial of degree <= D in t between kinks (D = 1: piecewise
  linear). Each cell is evaluated as a Taylor polynomial at tc just right of the cell start; the cell ends at the
  smallest positive kink root. Inside a cell the argmax can change only at roots of L_k - L_j (k = either target),
  so the cell is split at those roots (and at t = 0) and the argmax is read at each sub-interval's midpoint.
- numerical (net.mode = "num"): correctness on G nodes per side (midpoints of G equal-CDF bins), each change between
  neighbouring nodes located by B bisection steps; approximate (misses error slivers narrower than a bin that do not
  change the node values). Gaussian CDF only.
cost: FLOPs actually executed / FLOPs of one plain forward pass of the same net (FPE of that net).
"""
import numpy as np
from scipy.special import ndtr, ndtri

import netnp as N

SQ2 = np.sqrt(2.0)


class Net:
    def __init__(self, kind, w, mode=None):
        self.kind, self.w = kind, w
        self.mode = N.MODE[kind] if mode is None else mode
        self.homog = N.HOMOG[kind]
        self.fpe = N.forward(kind, w, np.zeros((1, 10)))[1]

    def logits_tp(self, y, d, tc, D):
        ctx = N.Ctx(len(y), D)
        L = N.FORWARD[self.kind](self.w, N.line_input(y, d, tc, D, ctx))
        return L.c, ctx


def all_roots(c, lo, hi):
    """c (K, D+1): real roots strictly inside (lo, hi) (per row), as (K, D) with nan elsewhere."""
    K, D1 = c.shape
    if D1 == 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (-c[:, 0] / c[:, 1])[:, None]
    else:
        r = np.full((K, D1 - 1), np.nan)
        scale = np.abs(c).max(1)
        nz = np.abs(c) > 1e-13 * scale[:, None]
        deg = np.where(nz.any(1), D1 - 1 - np.argmax(nz[:, ::-1], 1), 0)
        for d in range(1, D1):
            idx = np.nonzero(deg == d)[0]
            if not len(idx):
                continue
            cc = c[idx, :d + 1] / c[idx, d:d + 1]
            if d == 1:
                r[idx, 0] = -cc[:, 0]
                continue
            M = np.zeros((len(idx), d, d)); M[:, 1:, :-1] = np.eye(d - 1); M[:, :, -1] = -cc[:, :d]
            ev = np.linalg.eigvals(M)
            rr = np.where(np.abs(ev.imag) <= 1e-7 * (1 + np.abs(ev.real)), ev.real, np.nan)
            dc = cc[:, 1:] * np.arange(1, d + 1)
            for _ in range(2):
                f = N.poly_eval(cc[:, None, :], rr); fp = N.poly_eval(dc[:, None, :], rr)
                with np.errstate(divide="ignore", invalid="ignore"):
                    st = np.where(np.abs(fp) > 0, f / fp, 0.0)
                rr = rr - np.where(np.isfinite(st), st, 0.0)
            r[idx, :d] = rr
    lo = np.asarray(lo); hi = np.asarray(hi)
    ok = (r > lo[:, None]) & (r < hi[:, None]) & np.isfinite(r)
    return np.where(ok, r, np.nan)


def _dirs(y, a, b):
    n = len(y); rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    return d


def walk_exact(net, y, a, b, tau, kind, cdf=ndtr, eps=1e-10, max_cells=4000, lo=None):
    D = net.mode
    n, T = y.shape
    d = _dirs(y, a, b)
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    t = -tau.copy() if lo is None else np.asarray(lo, float).copy()
    mass = np.zeros(n); flops = np.zeros(n); cells = np.zeros(n, int)
    act = np.nonzero(t < tau)[0]
    it = 0
    while len(act) and it < max_cells:
        it += 1
        m = len(act)
        tc = t[act] + eps * (1 + np.abs(t[act]))
        L, ctx = net.logits_tp(y[act], d[act], tc, D)            # (m, T, D+1) in u = t - tc
        nxt = np.minimum(tc + ctx.kink, tau[act])
        ulo, uhi = t[act] - tc, nxt - tc
        rr = np.arange(m)
        brk = [np.where((-tc > ulo) & (-tc < uhi), -tc, np.nan)[:, None]]
        for k in (tneg[act], tpos[act]):
            diff = (L[rr, k][:, None, :] - L).reshape(m * T, D + 1)     # (m*T, D+1)
            brk.append(all_roots(diff, np.repeat(ulo, T), np.repeat(uhi, T)).reshape(m, -1))
        brk = np.sort(np.concatenate(brk, 1), 1)                 # nan last
        nb = brk.shape[1]
        edges = np.concatenate([ulo[:, None], brk, np.full((m, 1), np.nan)], 1)
        cnt = np.isfinite(brk).sum(1)
        edges[rr, cnt + 1] = uhi
        lo_e, hi_e = edges[:, :-1], edges[:, 1:]                 # (m, nb+1) sub-intervals, nan when unused
        valid = np.isfinite(lo_e) & np.isfinite(hi_e) & (hi_e > lo_e)
        mid = np.where(valid, 0.5 * (lo_e + hi_e), 0.0)
        Lm = N.poly_eval(L[:, None, :, :], mid[:, :, None])       # (m, nb+1, T)
        pick = Lm.argmax(2)
        targ = np.where(tc[:, None] + mid < 0, tneg[act][:, None], tpos[act][:, None])
        wrong = valid & (pick != targ)
        pm = np.where(wrong, cdf(np.where(valid, tc[:, None] + hi_e, 0.0)) - cdf(np.where(valid, tc[:, None] + lo_e, 0.0)), 0.0)
        mass[act] += pm.sum(1)
        # executed FLOPs: the Taylor forward + logit-difference roots + midpoint evaluations + CDFs
        flops[act] += ctx.flops / m + 2 * T * (2 if D == 1 else 10 * D ** 3) + 2 * (nb + 1) * T * (D + 1) + 40 * (nb + 1)
        cells[act] += 1
        t[act] = nxt
        act = act[nxt < tau[act]]
    return mass, flops / net.fpe, cells


def walk_num(net, y, a, b, tau, kind, cdf=ndtr, G=32, B=12, lo=None):
    assert cdf is ndtr, "numerical walk: Gaussian window only (non-homogeneous nets)"
    assert lo is None
    n, T = y.shape
    d = _dirs(y, a, b)
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    g = G // 2
    mass = np.zeros(n); nfwd = np.zeros(n)
    for side in (0, 1):
        plo, phi_ = (-tau, np.zeros(n)) if side == 0 else (np.zeros(n), tau)
        k = tneg if side == 0 else tpos
        qlo, qhi = cdf(plo), cdf(phi_)
        qn = qlo[:, None] + (np.arange(g) + 0.5)[None] / g * (qhi - qlo)[:, None]     # (n, g)
        tn = ndtri(qn)
        def correct(tt, rows):
            X = y[rows] + tt[:, None] * d[rows]
            Lg, _ = N.forward(net.kind, net.w, X)
            return Lg.argmax(1) == k[rows]
        rows = np.repeat(np.arange(n), g)
        ok = correct(tn.ravel(), rows).reshape(n, g)
        nfwd += g
        # bins: [qlo, q_0], [q_0, q_1], ..., [q_{g-1}, qhi]; node states known at q_i
        wrongq = np.where(~ok[:, 0], qn[:, 0] - qlo, 0.0) + np.where(~ok[:, -1], qhi - qn[:, -1], 0.0)
        same = ok[:, 1:] == ok[:, :-1]
        wrongq += np.where(same & ~ok[:, :-1], qn[:, 1:] - qn[:, :-1], 0.0).sum(1)
        ci, cj = np.nonzero(~same)
        if len(ci):
            l, h = tn[ci, cj].copy(), tn[ci, cj + 1].copy()
            okl = ok[ci, cj]
            for _ in range(B):
                mt = 0.5 * (l + h)
                om = correct(mt, ci)
                left = om == okl
                l = np.where(left, mt, l); h = np.where(left, h, mt)
            np.add.at(nfwd, ci, B)
            x = 0.5 * (l + h); qx = cdf(x)
            part = np.where(okl, qn[ci, cj + 1] - qx, qx - qn[ci, cj])
            np.add.at(wrongq, ci, part)
        mass += wrongq
    return mass, nfwd + 0.0 * mass, np.full(n, -1)


def walk(net, y, a, b, tau, kind, cdf=ndtr, **kw):
    if net.mode == "num":
        return walk_num(net, y, a, b, tau, kind, cdf=cdf, **kw)
    return walk_exact(net, y, a, b, tau, kind, cdf=cdf, **kw)

"""T455: rigorous upper bound on the slab mass of a homogeneity-reduced u-box (T414 coordinates, see
late_triple_verify/net.py). Exact representation (u_a, u_l integrated in closed form):
  M = int_{g >= t0} int_{r in [r0, r1]} F(r, g) dr dg,
  F = g/(2 pi) exp(-g^2 (r^2 + (r+1)^2) / 2) * [Phi(g(r+1+a1)) - Phi(g(r+1+a0))] * prod_k [Phi(g(r-l0_k)) - Phi(g(r-l1_k))].
Upper bound = sum over a grid of (r, g) cells of area * (product of per-factor sups over the cell)
  + the part g > GMAX, bounded by the radial formula (vol * 16 (2 pi)^-5 m^-5 Gamma(5, m GMAX^2 / 2), m = min |y|^2).
Per Phi difference over a cell, with z_lo(r, g) <= z_hi(r, g) its two arguments (bilinear in (r, g)):
  sup <= min( Phi(max z_hi) - Phi(min z_lo),  g_hi * width * max_{[min z_lo, max z_hi]} phi ).
All factors are nonnegative, so the product of sups bounds the sup of the product. Every term is monotone in the cell
corners, so it is evaluated at corners only.
Two implementations of the same sum: `upper` (float64, vectorised; final value inflated by (1 + 1e-9) + 1e-300 to cover
rounding, since every elementary op errs by a few ulp at most) and `upper_arb` (python-flint arb, outward-rounded ball
arithmetic, slow; used to validate `upper` on samples: upper >= arb's upper endpoint)."""
import math

import numpy as np
import torch as th

T0, GMAX = 2.5, 20.0
SQ2 = math.sqrt(2.0)
INFL = 1 + 1e-9
dt = th.float64


def _Phi(z):
    return 0.5 * th.special.erfc(-z / SQ2)


def _phimax(zl, zh):
    """max of phi on [zl, zh]"""
    m = th.where(zl > 0, zl, th.where(zh < 0, zh, th.zeros_like(zl)))
    return th.exp(-0.5 * m * m) / math.sqrt(2 * math.pi)


def ggrid(ng):
    s = th.linspace(0, 1, ng + 1, dtype=dt); return T0 + (GMAX - T0) * s ** 2


def _diff_sup(cl, ch, rl, rh, gl, gh):
    """sup over the cell [rl, rh] x [gl, gh] of Phi(g (r + ch)) - Phi(g (r + cl)), cl <= ch; shapes broadcast"""
    # z = g (r + c): extremes at corners (bilinear)
    zh = th.maximum(th.maximum(gl * (rl + ch), gl * (rh + ch)), th.maximum(gh * (rl + ch), gh * (rh + ch)))
    zl = th.minimum(th.minimum(gl * (rl + cl), gl * (rh + cl)), th.minimum(gh * (rl + cl), gh * (rh + cl)))
    # Phi(zh) - Phi(zl) without cancellation: in the upper tail as Q(zl) - Q(zh) (Q = 1 - Phi); a slack of 1e-14 x the
    # larger term covers the few-ulp errors of erfc and the subtraction (V1 found 0s from 1 - 1 cancellation otherwise)
    up = zl >= 0
    big = th.where(up, _Phi(-zl), _Phi(zh)); small = th.where(up, _Phi(-zh), _Phi(zl))
    b1 = big - small + 1e-14 * big
    b2 = gh * (ch - cl) * _phimax(zl, zh)
    return th.minimum(b1, b2).clamp(min=0)


def upper(lo, hi, nr=24, ng=48, m=None):
    """rigorous (float, inflated) upper bound on the slab mass of each u-box; lo, hi (N, 9); m = lower bound of
    min |y|^2 over the box (for the g > GMAX tail; 0.5 always valid)"""
    N = lo.shape[0]
    re = lo[:, :1] + (hi[:, :1] - lo[:, :1]) * th.linspace(0, 1, nr + 1, dtype=dt)[None]       # (N, nr+1)
    rl = re[:, :-1, None]; rh = re[:, 1:, None]                                                  # (N, nr, 1)
    ge = ggrid(ng); gl = ge[:-1][None, None]; gh = ge[1:][None, None]                            # (1, 1, ng)
    # Gaussian factor: q = r^2 + (r+1)^2 = 2 (r + 1/2)^2 + 1/2, min over the r-cell
    rc = th.minimum(th.maximum(th.full_like(rl, -0.5), rl), rh); qmin = 2 * (rc + 0.5) ** 2 + 0.5
    f = gh / (2 * math.pi) * th.exp(-0.5 * gl * gl * qmin)
    # u_a: Phi(g(r+1+a1)) - Phi(g(r+1+a0))
    f = f * _diff_sup(1 + lo[:, 1, None, None], 1 + hi[:, 1, None, None], rl, rh, gl, gh)
    # lows: Phi(g(r - l0)) - Phi(g(r - l1)) = Phi(g(r + c_h)) - Phi(g(r + c_l)) with c_h = -l0, c_l = -l1
    for k in range(7):
        f = f * _diff_sup(-hi[:, 2 + k, None, None], -lo[:, 2 + k, None, None], rl, rh, gl, gh)
    area = (rh - rl) * (gh - gl)
    body = (f * area).sum((1, 2))
    if m is None:
        m = th.full((N,), 0.5, dtype=dt)
    w = m * GMAX ** 2 / 2
    tail = (hi - lo).prod(1) * 16 * (2 * math.pi) ** -5 * m ** -5 * 24 * th.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
    return (body + tail) * INFL + 1e-300


def upper_arb(lo, hi, nr=24, ng=48, m=0.5, prec=80):
    """the same sum in arb ball arithmetic for ONE box (lists of 9 floats); returns the arb ball's upper endpoint as
    a float rounded up. Grid points are the same float values as in `upper` (any grid gives a valid bound)."""
    from flint import arb, ctx
    ctx.prec = prec
    A = lambda v: arb(float(v))                                                               # exact conversion
    pi2 = 2 * arb.pi(); sq2 = arb(2).sqrt(); sqpi2 = pi2.sqrt()
    Phi = lambda z: (-z / sq2).erfc() / 2

    def mx(*v):
        r = v[0]
        for x in v[1:]:
            r = r.max(x)
        return r

    def mn(*v):
        r = v[0]
        for x in v[1:]:
            r = r.min(x)
        return r

    def phimax(zl, zh):
        # zl, zh balls; max phi on [zl, zh] <= phi(dist of 0 to the interval), using outward lower dist
        if zl > 0:
            d = zl
        elif zh < 0:
            d = -zh
        elif (zl <= 0) and (zh >= 0):
            d = arb(0)
        else:                                                                                  # undecided sign: 0 is safe
            d = arb(0)
        return (-(d * d) / 2).exp() / sqpi2

    def dsup(cl, ch, rl, rh, gl, gh):
        zh = mx(gl * (rl + ch), gl * (rh + ch), gh * (rl + ch), gh * (rh + ch))
        zl = mn(gl * (rl + cl), gl * (rh + cl), gh * (rl + cl), gh * (rh + cl))
        if zl > 0:                                                                             # upper tail, no cancellation
            b1 = ((zl / sq2).erfc() - (zh / sq2).erfc()) / 2
        else:
            b1 = Phi(zh) - Phi(zl)
        b2 = gh * (ch - cl) * phimax(zl, zh)
        b = b1.min(b2)
        return b.max(arb(0))

    r0, r1 = float(lo[0]), float(hi[0])
    re = [r0 + (r1 - r0) * (i / nr) for i in range(nr + 1)]                                 # float grid (as in upper up to rounding)
    re[0], re[-1] = r0, r1
    ge = ggrid(ng).tolist()
    tot = arb(0)
    for i in range(nr):
        rl, rh = A(re[i]), A(re[i + 1]); rc = mn(mx(arb(-0.5), rl), rh); qmin = 2 * (rc + arb(0.5)) ** 2 + arb(0.5)
        for j in range(ng):
            gl, gh = A(ge[j]), A(ge[j + 1])
            f = gh / pi2 * (-(gl * gl) * qmin / 2).exp()
            f = f * dsup(1 + A(lo[1]), 1 + A(hi[1]), rl, rh, gl, gh)
            for k in range(7):
                f = f * dsup(-A(hi[2 + k]), -A(lo[2 + k]), rl, rh, gl, gh)
            tot = tot + f * (rh - rl) * (gh - gl)
    vol = arb(1)
    for d in range(9):
        vol = vol * (A(hi[d]) - A(lo[d]))
    mm = A(m); w = mm * GMAX ** 2 / 2
    tail = vol * 16 / pi2 ** 5 / mm ** 5 * 24 * (-w).exp() * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
    tot = tot + tail
    return float(tot.upper()) if hasattr(tot, "upper") else float(tot.mid() + tot.rad())

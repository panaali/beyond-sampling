"""Direct (rejection-free) model-free samplers for surface points and zone inputs.

Surface A: ybar has density ∝ phi(v; var 1/2) * 8 sf(v) cdf(v)^7 (tabulated inverse CDF, model-free); one other
input is N(0,1) truncated above ybar, seven are truncated below; positions uniformly random.
Surface B: ybar density ∝ phi(v; 1/2) cdf(v)^8; all eight others truncated below ybar.
Blind: at most one positive input: P(k=0)=1/11, P(k=1)=10/11 given blind.
Each generated length-10 vector costs ~0.04 forward-pass equivalents (normals + sort, ~250 flops vs ~6e3).
"""
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

T = 10
INPUT_COST = 0.04
_v = np.linspace(-7, 7, 200001)
_s = np.sqrt(0.5)


def _inv(dens):
    c = np.cumsum(dens); c = (c - c[0]) / (c[-1] - c[0])
    return lambda u: np.interp(u, c, _v)


_invA = _inv(norm.pdf(_v, scale=_s) * norm.sf(_v) * norm.cdf(_v) ** 7)
_invB = _inv(norm.pdf(_v, scale=_s) * norm.cdf(_v) ** 8)


def _trunc_below(c, shape, rng):     # N(0,1) | X < c
    return ndtri(np.clip(rng.random(shape) * ndtr(c), 1e-300, 1 - 1e-16))


def _trunc_above(c, shape, rng):     # N(0,1) | X > c
    p = ndtr(c); return ndtri(np.clip(p + rng.random(shape) * (1 - p), 1e-300, 1 - 1e-16))


def surface(kind, n, rng):
    yb = (_invA if kind == "A" else _invB)(rng.random(n))
    oth = _trunc_below(yb[:, None], (n, 8), rng)
    if kind == "A":
        oth[:, 0] = _trunc_above(yb, n, rng)
    perm = np.argsort(rng.random((n, T)), 1)          # random positions: first two slots are the tied pair
    x = np.empty((n, T)); rows = np.arange(n)[:, None]
    vals = np.column_stack([yb, yb, oth])
    x[rows, perm] = vals
    a, b = perm[:, 0], perm[:, 1]
    os_ = -np.sort(-oth, 1)
    return x, a, b, yb, os_


def blind(n, rng):
    k1 = rng.random(n) < 10 / 11
    x = -np.abs(rng.standard_normal((n, T)))          # all negative
    pos = rng.integers(0, T, n)
    r = np.nonzero(k1)[0]
    x[r, pos[r]] = np.abs(rng.standard_normal(len(r)))
    return x

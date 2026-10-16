"""Psychometric-observer estimator core: model weights -> exact local linear maps -> Gaussian 'internal noise'."""
import itertools
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

T = 10
phi = lambda z: np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
TRIPLES = np.array([p for p in itertools.permutations(range(T), 3)])   # (720,3) positions of rank1,2,3


def forward_gates(model, x):
    """x (N,T) -> logits (N,T), gates (N,T,H)."""
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0])); G = []
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); G.append(h > 0)
    return h @ Wo.T, np.stack(G, 1)


def logit_map(model, G):
    """Exact map A (N,10 logits,10 inputs) for given gate patterns: logits = A @ x inside the cone."""
    Wi, Wh, Wo = model
    N = len(G); J = np.zeros((N, Wh.shape[0], T))
    for t in range(T):
        J = G[:, t][:, :, None] * np.einsum("ij,njt->nit", Wh, J)
        J[:, :, t] += G[:, t] * Wi[None]
    return np.einsum("ki,nit->nkt", Wo, J)


def trunc_moments(c):
    """Mean and variance of N(0,1) truncated to (-inf, c)."""
    lam = phi(c) / np.maximum(ndtr(c), 1e-300)
    return -lam, 1 - c * lam - lam ** 2


def trunc_quantiles(c, k):
    """k representative values of N(0,1) | < c (mid-quantiles), descending."""
    u = (np.arange(k) + 0.5) / k
    return norm.ppf(ndtr(c) * u)[::-1]


def bin_moments(c, q):
    """Mean and variance of N(0,1)|<c restricted to each of q equal-mass bins. c: array (M,) -> (q,M),(q,M)."""
    c = np.asarray(c, float); Pc = ndtr(c)
    edges = [norm.ppf(np.clip(Pc * r / q, 1e-300, 1)) for r in range(q + 1)]
    edges[0] = np.full_like(c, -np.inf); edges[-1] = c
    M = []; V = []
    for r in range(q):
        lo, hi = edges[r], edges[r + 1]
        pl = np.where(np.isinf(lo), 0, phi(np.where(np.isinf(lo), 0, lo))); ph = phi(hi)
        zl = np.where(np.isinf(lo), 0, np.where(np.isinf(lo), 0, lo) * pl); zh = hi * ph
        dP = Pc / q
        m = (pl - ph) / dP; e2 = 1 + (zl - zh) / dP
        M.append(m); V.append(np.maximum(e2 - m * m, 0))
    return np.array(M), np.array(V)


def bin_medians(c, q):
    return norm.ppf(ndtr(c) * (np.arange(q) + 0.5) / q)


def map_and_cone(model, x):
    """Exact local logit map A (N,10,10) plus, for each input coordinate t, the interval [lo_t, hi_t] of
    x_t (others fixed) over which every ReLU gate keeps its sign, i.e. where A stays exact."""
    Wi, Wh, Wo = model
    N = len(x); H = Wh.shape[0]
    h = np.zeros((N, H)); J = np.zeros((N, H, T))
    lo = np.full((N, T), -np.inf); hi = np.full((N, T), np.inf)
    for s in range(T):
        z = h @ Wh.T + x[:, s:s + 1] * Wi[None]                  # pre-activations (N,H)
        P = np.einsum("ij,njt->nit", Wh, J); P[:, :, s] += Wi[None]   # dz/dx (N,H,T)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = -z[:, :, None] / P                               # step along x_t to flip gate i
        delta = np.where(np.isfinite(delta), delta, np.nan)
        pos = np.where(delta > 0, delta, np.inf).min(1); neg = np.where(delta < 0, delta, -np.inf).max(1)
        hi = np.minimum(hi, x + pos); lo = np.maximum(lo, x + neg)
        g = z > 0
        h = np.where(g, z, 0); J = g[:, :, None] * P
    return h @ Wo.T, np.einsum("ki,nit->nkt", Wo, J), lo, hi


def clipped_bin_moments(c, q, r, lo, hi, n=24):
    """E and Var of clip(x, lo, hi) for x ~ N(0,1)|<c restricted to equal-mass bin r (scalar c; lo,hi arrays)."""
    Pc = ndtr(c)
    u = (r + (np.arange(n) + 0.5) / n) / q                  # mid-quantile rule inside the bin
    xs = norm.ppf(Pc * u)                                   # (n,)
    v = np.clip(xs[None, :], lo[:, None], hi[:, None])
    return v.mean(1), v.var(1)

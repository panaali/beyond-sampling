"""T405 own truth code (written for T405, shares nothing with T364's truthc / T372's truth_fd2).
Exact conditional law of x ~ N(0, I_10) given g23 = v2 - v3 > t0 (v1 > v2 > v3 > ... the order statistics):
  positions: a uniform random permutation (top, 2nd, the 8 others);
  v2 has density f(v) = 90 phi(v) (1 - Phi(v)) Phi(v - t0)^8 / p_gap  (p_gap = P(g23 > t0), its integral);
  given v2: v1 ~ N(0,1) truncated to (v2, inf); the 8 others iid N(0,1) truncated to (-inf, v2 - t0).
v2 is drawn by inverse CDF of f on a fine model-free grid (2e5 cells on [-7, 9], trapezoid CDF, linear interpolation).
Batched ReLU-RNN forward over a stack of weight sets: h_t = ReLU(Wh h_{t-1} + Wi x_t), logits = Wo h_T."""
import numpy as np
from scipy.special import ndtr, ndtri

GRID = np.linspace(-7.0, 9.0, 200001)


def _phi(v):
    return np.exp(-0.5 * v * v) / np.sqrt(2 * np.pi)


def v2_table(t0):
    f = 90 * _phi(GRID) * ndtr(-GRID) * ndtr(GRID - t0) ** 8
    c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))])
    return c[-1], c / c[-1]


def draw(rng, n, t0, cdf):
    """n draws of (x (n, 10), position of the rank-3 input (n,))."""
    v2 = np.interp(rng.random(n), cdf, GRID)
    v1 = -ndtri(rng.random(n) * ndtr(-v2))                       # N truncated to (v2, inf)
    lo = ndtri(rng.random((n, 8)) * ndtr(v2 - t0)[:, None])      # 8 iid N truncated to (-inf, v2 - t0)
    vals = np.concatenate([v1[:, None], v2[:, None], lo], 1)      # columns: rank 1, rank 2, then the 8 lows
    perm = np.argsort(rng.random((n, 10)), 1)                     # perm[:, j] = position of column j
    x = np.empty((n, 10)); np.put_along_axis(x, perm, vals, 1)
    p3 = perm[np.arange(n), 2 + lo.argmax(1)]
    return x, p3


def stack(models):
    return tuple(np.stack([m[k] for m in models]) for k in range(3))


def argmax_stack(S, x):
    """S = (Wi (M, H), Wh (M, H, H), Wo (M, C, H)); x (n, T) -> argmax logits (M, n). Per-model loop, in-place."""
    Wi, Wh, Wo = S; M, H = Wi.shape; n, T = x.shape; out = np.empty((M, n), int)
    h = np.empty((n, H)); buf = np.empty((n, H))
    for m in range(M):
        WhT = np.ascontiguousarray(Wh[m].T); np.multiply(x[:, :1], Wi[m], out=h); np.maximum(h, 0.0, out=h)
        for t in range(1, T):
            np.matmul(h, WhT, out=buf); buf += x[:, t:t + 1] * Wi[m]; np.maximum(buf, 0.0, out=h)
        out[m] = (h @ Wo[m].T).argmax(1)
    return out

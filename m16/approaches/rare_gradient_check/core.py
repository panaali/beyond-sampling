"""T385 core (own code, written without T372's estimator): models, the exact conditional law of x given g23 > t0,
the RNN forward pass, and the event E1(t0) = {argmax of logits = position of the rank-3 input, v2 - v3 > t0}.
Weight vector convention (same as T372 so numbers can be compared): concat(Wi (16), Wh (16x16 row-major,
Wh[i, j] = weight from neuron j into neuron i), Wo (10x16 row-major))."""
import os

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEV3 = os.path.join(ROOT, "data", "t385", "dev3_weights.npz")   # = origin/claude/t286-cheap-frontier rare_events/ext/dev3
DEV3_SHA = "62c952037e8c9ec214ee96b3dc43923a9e19cb5f137dbf16b7f6308d721591f9"
SHAPES = [(16,), (16, 16), (10, 16)]
CASES = {"orig": 1.25, "dev3_s528": 2.75}


def load(name):
    if name == "orig":
        from alg_zoo import example_2nd_argmax
        m = example_2nd_argmax().eval()
        return (m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float),
                m.linear.weight.detach().numpy().astype(float))
    import hashlib
    assert hashlib.sha256(open(DEV3, "rb").read()).hexdigest() == DEV3_SHA
    z = np.load(DEV3); i = list(map(str, z["names"])).index(name)
    return z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float)


def flat(model):
    return np.concatenate([np.ravel(w) for w in model])


def unflat(v):
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s)); out.append(np.asarray(v[i:i + n], float).reshape(s)); i += n
    return tuple(out)


def logits(model, X):
    Wi, Wh, Wo = model; WhT = Wh.T.copy(); h = np.zeros((len(X), len(Wi)))
    for t in range(X.shape[1]):
        h = h @ WhT; h += X[:, t:t + 1] * Wi; np.maximum(h, 0.0, out=h)
    return h @ Wo.T


class CondLaw:
    """Exact law of x ~ N(0, I_10) given v2 - v3 > t0 (v_r = r-th largest). Marginal of v2:
    f(v) = 90 phi(v) Phibar(v) Phi(v - t0)^8 (normalised by p_gap = P(g23 > t0)), tabulated on a fine grid and inverted;
    v1 | v2 ~ N truncated to (v2, inf); v3 = max of 8 iid N truncated below v2 - t0; the 7 lows iid N below v3."""

    def __init__(self, t0, lo=-8.0, hi=11.0, n=1_500_001):
        self.t0 = t0; g = np.linspace(lo, hi, n)
        logf = np.log(90.0) - g ** 2 / 2 - 0.5 * np.log(2 * np.pi) + log_ndtr(-g) + 8 * log_ndtr(g - t0)
        f = np.exp(logf); h = g[1] - g[0]
        # Simpson on pairs of intervals for p_gap; trapezoid cumulative for the inverse CDF
        self.p_gap = float(h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()))
        c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) * h / 2)]); self.cdf = c / c[-1]; self.grid = g

    def draw(self, rng, n):
        """-> x (n, 10), rank_pos (n, 10): rank_pos[:, r] = position of the (r+1)-th largest value."""
        v2 = np.interp(rng.random(n), self.cdf, self.grid)
        v1 = -ndtri(rng.random(n) * ndtr(-v2))
        v3 = ndtri(np.exp(log_ndtr(v2 - self.t0) + np.log(rng.random(n)) / 8))
        lows = -np.sort(-ndtri(np.exp(log_ndtr(v3)[:, None] + np.log(rng.random((n, 7))))), 1)
        vals = np.column_stack([v1, v2, v3, lows])
        pos = np.argsort(rng.random((n, 10)), 1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], pos] = vals
        return x, pos

"""Independent helpers for the T79 review of E31 (written from scratch; does NOT import stoch_margin/sm.py).

Conventions: model = (Wi (H,), Wh (H,H), Wo (T,H)); input x (N, T); label = index of the 2nd largest entry.
Skeleton of order K = positions (in rank order) and values of the K largest inputs. Given the skeleton, the other
T-K inputs are iid N(0,1) conditioned on being < v_K (sampled here with scipy.stats.truncnorm, not inverse-CDF).
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
from scipy.stats import truncnorm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "m16"))
T = 10
OUT = os.path.join(ROOT, "data", "review_stoch_margin")


def dev_model(name):
    import leaderboard as LB
    return LB.models(dev=True)[name]


def run(model, x, keep_pre=False):
    """Forward pass. Returns logits (N,T) and optionally all pre-activations (N,T,H)."""
    Wi, Wh, Wo = model
    N = x.shape[0]
    h = np.zeros((N, Wh.shape[0]))
    pres = np.empty((N, T, Wh.shape[0])) if keep_pre else None
    WhT = Wh.T
    for t in range(T):
        z = h @ WhT + np.outer(x[:, t], Wi)
        if keep_pre:
            pres[:, t] = z
        h = np.maximum(z, 0.0)
    return h @ Wo.T, pres


def jacobian(model, pre):
    """Analytic d logits / d x_t inside the gate pattern of the given pre-activations. pre (N,T,H) -> (N,T_out,T)."""
    Wi, Wh, Wo = model
    N = pre.shape[0]
    D = (pre > 0).astype(float)                       # (N,T,H)
    G = np.repeat(Wo[None], N, 0)                      # (N,10,H)  d L / d h_s
    J = np.empty((N, Wo.shape[0], T))
    for s in range(T - 1, -1, -1):
        Gz = G * D[:, s][:, None, :]                   # d L / d z_s
        J[:, :, s] = Gz @ Wi
        G = Gz @ Wh                                    # d L / d h_{s-1}
    return J


def skeletons(n, K, rng):
    x = rng.standard_normal((n, T))
    order = np.argsort(-x, axis=1)
    pos = order[:, :K]
    return pos, np.take_along_axis(x, pos, 1)


def low_draws(vK, m, nl, rng):
    """(n,) thresholds -> (n,m,nl) iid N(0,1) | < vK via truncnorm."""
    b = np.broadcast_to(vK[:, None, None], (len(vK), m, nl))
    return truncnorm.rvs(-np.inf, b, size=b.shape, random_state=rng)


def free_positions(pos):
    n, K = pos.shape
    mask = np.ones((n, T), bool)
    mask[np.arange(n)[:, None], pos] = False
    return np.nonzero(mask)[1].reshape(n, T - K)


def assemble(pos, val, free, low):
    """-> x (n,m,T)."""
    n, m, nl = low.shape
    x = np.zeros((n, m, T))
    r = np.arange(n)[:, None, None]
    x[r, np.arange(m)[None, :, None], pos[:, None, :]] = val[:, None, :]
    x[r, np.arange(m)[None, :, None], free[:, None, :]] = low
    return x


def cond_mean(vK):
    from scipy.stats import norm
    return -norm.pdf(vK) / norm.cdf(vK)


def summarize(q, qred, nsk):
    d = qred - q
    return dict(bias_pp=float(100 * d.mean()), se_pp=float(100 * d.std(ddof=1) / np.sqrt(nsk)),
                abs_pp=float(100 * np.abs(d).mean()))

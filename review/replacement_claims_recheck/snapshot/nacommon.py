"""Shared helpers for T241 (neuron account). Exact float64 forward of M_16,10.

Weights are read from m16_weights.npz, exported from ARC's public checkpoint
gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth (sha256 498bd9a8...),
identical to alg_zoo.example_2nd_argmax().
"""
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_w = np.load(os.path.join(_here, "m16_weights.npz"))
Wi, Wh, Wo = _w["Wi"], _w["Wh"], _w["Wo"]
H, T = 16, 10


def sample(n, seed):
    return np.random.default_rng(seed).standard_normal((n, T))


def run(x):
    """Hidden states (N, T+1, H), hs[:, t] = h_t, h_0 = 0; also pre-activations (N, T+1, H) (pre[:,0]=0)."""
    N = len(x)
    hs = np.zeros((N, T + 1, H))
    pre = np.zeros((N, T + 1, H))
    for t in range(T):
        pre[:, t + 1] = hs[:, t] @ Wh.T + x[:, t:t + 1] * Wi[None]
        hs[:, t + 1] = np.maximum(pre[:, t + 1], 0)
    return hs, pre


def run_from(h, x, t0):
    """Continue from h = h_{t0} through step T; return final logits."""
    for t in range(t0, T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def target(x):
    return np.argsort(x, 1)[:, -2]


def cases(x):
    """Boolean masks: x9 is max, x9 is 2nd, x9 below both; blind = at most one positive input."""
    r = np.argsort(np.argsort(-x, 1), 1)[:, 9]  # rank of x9 (0 = max)
    return {"x9max": r == 0, "x9second": r == 1, "x9below": r >= 2,
            "blind": (x > 0).sum(1) <= 1}


def metrics(logits, x, ref_pred):
    pred = logits.argmax(1)
    y = target(x)
    ok = pred == y
    out = {"acc": ok.mean(), "agree": (pred == ref_pred).mean()}
    for k, m in cases(x).items():
        out["acc_" + k] = ok[m].mean()
    return out


def order_stats(x):
    """Running quantities per step t=1..T (arrays (N, T+1), index t = after reading x_{t-1}).
    m1,m2,m3: running top-3 values (-inf->nan when undefined); p1,p2: positions; age1, age2: steps since."""
    N = len(x)
    m = np.full((N, T + 1, 3), np.nan)
    p = np.full((N, T + 1, 3), -1)
    for t in range(1, T + 1):
        xs = x[:, :t]
        k = min(3, t)
        idx = np.argsort(-xs, 1)[:, :k]
        m[:, t, :k] = np.take_along_axis(xs, idx, 1)
        p[:, t, :k] = idx
    return m, p

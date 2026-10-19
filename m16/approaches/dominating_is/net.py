"""M_16,10 (alg_zoo.example_2nd_argmax) forward pass and the far-gap event E1(t0), as in T400 prob_bound.py:
E1(t0) = {argmax of step-10 logits = position of x_(3)} and x_(2) - x_(3) >= t0, x ~ N(0, I_10).
Triple = (position of x_(1), of x_(2), of x_(3)) (T400 convention: sorted descending)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from common import Wi, Wh, Wo, H, T


def logits(X):
    h = np.zeros((len(X), H))
    for t in range(T):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def top3(X):
    top = np.argpartition(-X, 3, axis=1)[:, :3]
    v = np.take_along_axis(X, top, 1)
    o = np.argsort(-v, 1)
    return np.take_along_axis(top, o, 1), np.take_along_axis(v, o, 1)


def event(X, t0):
    """returns (hit bool array, triple array, n_forward) ; the network runs only on draws with g23 >= t0."""
    top, v = top3(X)
    sel = v[:, 1] - v[:, 2] >= t0
    hit = np.zeros(len(X), bool)
    if sel.any():
        hit[sel] = logits(X[sel]).argmax(1) == top[sel, 2]
    return hit, top, int(sel.sum())

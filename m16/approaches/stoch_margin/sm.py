"""Shared helpers for stoch_margin (T1). Run scripts from the repo root with OMP_NUM_THREADS=1."""
import os
import sys

import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, "m16")
T = 10


def get_models(names):
    """Named models: 'orig', 'seedK' (test, diagnosis only), 'dev_s<sigma>_<k>' (DEV noisy copies)."""
    import leaderboard as LB
    out = {}
    dev = None
    for n in names:
        if n.startswith("dev_"):
            if dev is None:
                dev = LB.models(dev=True)
            out[n] = dev[n]
        elif n == "orig":
            from arc_eval import Wi0, Wh0, Wo0
            out[n] = (Wi0, Wh0, Wo0)
        elif n.startswith("seed"):
            out[n] = LB._zoo(16, 10, int(n[4:]))
    return out


def logits(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def correct_from_logits(L, p2):
    return L.argmax(1) == p2


def sample_skeletons(n, rng):
    """Diagnosis only: skeletons from the exact law (top-3 positions and values of N(0,I_10))."""
    x = rng.standard_normal((n, T))
    o = np.argsort(-x, 1)
    pos = o[:, :3]
    val = np.take_along_axis(x, pos, 1)
    return pos, val


def trunc_below(v3, u):
    """Inverse-CDF of N(0,1) conditioned on X < v3; u in (0,1)."""
    return ndtri(u * ndtr(v3))


def build_inputs(pos, val, low):
    """pos (n,3), val (n,3), low (n,m,7) -> x (n,m,10) with low values in the free positions (ascending)."""
    n, m, _ = low.shape
    x = np.empty((n, m, T))
    mask = np.ones((n, T), bool)
    mask[np.arange(n)[:, None], pos] = False
    free = np.nonzero(mask)[1].reshape(n, 7)
    for k in range(3):
        x[np.arange(n), :, pos[:, k]] = val[:, k][:, None]
    for j in range(7):
        x[np.arange(n), :, free[:, j]] = low[:, :, j]
    return x, free


def sample_skeletons_k(n, K, rng):
    x = rng.standard_normal((n, T))
    pos = np.argsort(-x, 1)[:, :K]
    return pos, np.take_along_axis(x, pos, 1)


def build_inputs_k(pos, val, low):
    """pos (n,K), val (n,K), low (n,m,10-K)."""
    n, m, nl = low.shape
    K = pos.shape[1]
    x = np.empty((n, m, T))
    mask = np.ones((n, T), bool)
    mask[np.arange(n)[:, None], pos] = False
    free = np.nonzero(mask)[1].reshape(n, nl)
    for k in range(K):
        x[np.arange(n), :, pos[:, k]] = val[:, k][:, None]
    for j in range(nl):
        x[np.arange(n), :, free[:, j]] = low[:, :, j]
    return x, free

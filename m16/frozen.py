"""Gated-linear view: run the RNN with ReLU gates taken from another input (exactly linear in x given gates)."""
import numpy as np
from common import Wi, Wh, Wo, run


def gates(x):
    return run(x)[:, 1:] > 0          # (N, T, H)


def run_frozen(x, G):
    """h_{t+1} = G_t * (Wh h_t + Wi x_t): exact network iff G == gates(x)."""
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = G[:, t] * (h @ Wh.T + x[:, t:t + 1] * Wi[None])
    return h


def canonical(x, n_ref=200000, seed=99):
    """Same rank order as x, values replaced by expected order statistics of 10 N(0,1) (rank among x only)."""
    ref = np.sort(np.random.default_rng(seed).standard_normal((n_ref, x.shape[1])), 1).mean(0)
    ranks = np.argsort(np.argsort(x, 1), 1)
    return ref[ranks]

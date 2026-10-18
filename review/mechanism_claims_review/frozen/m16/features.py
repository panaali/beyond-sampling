"""Interpretable features of an input sequence x (N, T)."""
import numpy as np


def loo_max(x):
    """L[:, k] = max(0, x_j for j != k)."""
    N, T = x.shape
    xp = np.maximum(x, 0)
    s = np.sort(xp, 1)
    top, second = s[:, -1], s[:, -2]
    is_top = xp == top[:, None]
    return np.where(is_top, second[:, None], top[:, None])


def prefix_max(x):
    """P[:, t] = max(0, x_0..x_t)."""
    return np.maximum.accumulate(np.maximum(x, 0), 1)


def dictionary(x):
    """Named feature columns for regression."""
    N, T = x.shape
    L = loo_max(x)
    P = prefix_max(x)
    cols, names = [], []
    for j in range(T):
        cols += [x[:, j], np.maximum(x[:, j], 0), L[:, j], P[:, j]]
        names += [f"x{j}", f"r{j}", f"L{j}", f"P{j}"]
    return np.stack(cols, 1), names

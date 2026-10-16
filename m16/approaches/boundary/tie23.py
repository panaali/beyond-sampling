"""Conditional MC along the 2nd-vs-3rd tie direction (boundary-layer integration, exact in the normal coordinate)."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from seg import walk

SQ2 = np.sqrt(2.0)


def segments_23(x):
    """For each x: line through x along (e_r2 - e_r3)/sqrt2 restricted to the rank-preserving segment."""
    N = len(x); o = np.argsort(-x, 1); rows = np.arange(N)
    r1, a, b, r4 = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    xa, xb = x[rows, a], x[rows, b]
    d = np.zeros_like(x); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    t0 = (xa - xb) / SQ2
    x0 = x - t0[:, None] * d
    ybar = (xa + xb) / 2
    Tm = SQ2 * np.minimum(x[rows, r1] - ybar, ybar - x[rows, r4])
    return x0, d, -Tm, Tm, b, a, ybar


def cond_acc_23(model, x):
    x0, d, lo, hi, tn, tp, _ = segments_23(x)
    c, tot, p = walk(model, x0, d, lo, hi, tn, tp)
    return c / tot, p

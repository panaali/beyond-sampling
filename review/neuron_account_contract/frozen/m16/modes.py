"""Eigen-modes of the recurrence under the post-record steady gate set."""
import numpy as np
from common import Wh, Wo

STEADY = np.array([0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15])


def modes():
    M = Wh[np.ix_(STEADY, STEADY)]
    ev, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(ev))
    ev, V = ev[order], V[:, order]
    Vinv = np.linalg.inv(V)
    return ev, V, Vinv


def mode_coords(h):
    """Complex coordinates of hidden states h (N,16) in the eigenbasis (restricted to STEADY)."""
    ev, V, Vinv = modes()
    return h[:, STEADY] @ Vinv.T, ev

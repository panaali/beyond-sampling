"""Exact per-input linear map: given an input's own ReLU gates, logits = A(x) @ x with A computed from weights."""
import numpy as np
from common import Wi, Wh, Wo, run


def contribution_map(x):
    """J[n, :, t] = contribution of x_t to h_10 for input n (exact: h_10 = J @ x)."""
    G = run(x)[:, 1:] > 0
    N, T = x.shape
    J = np.zeros((N, Wh.shape[0], T))
    for t in range(T):
        J = G[:, t][:, :, None] * np.einsum("ij,njt->nit", Wh, J)
        J[:, :, t] += G[:, t] * Wi[None]
    return J


def logit_map(x):
    return np.einsum("ki,nit->nkt", Wo, contribution_map(x))   # (N, 10 logits, 10 inputs)

"""Shared helpers for analysing AlgZoo M_16,10 (example_2nd_argmax)."""
import numpy as np
import torch
from alg_zoo import example_2nd_argmax

_m = example_2nd_argmax().eval()
Wi = _m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(np.float64)   # (16,)
Wh = _m.rnn.weight_hh_l0.detach().numpy().astype(np.float64)        # (16,16)
Wo = _m.linear.weight.detach().numpy().astype(np.float64)           # (10,16)
H, T = Wh.shape[0], Wo.shape[0]


def standardized():
    """Rescale each neuron by |Wi| (ReLU is positively homogeneous), so Wi in {+-1}. Exactly equivalent."""
    s = np.abs(Wi)
    s[s < 1e-6] = 1.0
    return Wi / s, Wh * s[None, :] / s[:, None], Wo * s[None, :], s


def run(x, Wi=Wi, Wh=Wh):
    """All hidden states. x: (N,T) -> (N,T+1,H) with h_0 = 0."""
    N = len(x)
    hs = np.zeros((N, x.shape[1] + 1, len(Wi)))
    for t in range(x.shape[1]):
        hs[:, t + 1] = np.maximum(hs[:, t] @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return hs


def sample(n, seed):
    return np.random.default_rng(seed).standard_normal((n, T))


def target(x):
    return np.argsort(x, 1)[:, -2]


def model_acc(x, Wi=Wi, Wh=Wh, Wo=Wo):
    return float(((run(x, Wi, Wh)[:, -1] @ Wo.T).argmax(1) == target(x)).mean())

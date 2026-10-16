"""Psychometric-observer estimator core: model weights -> exact local linear maps -> Gaussian 'internal noise'."""
import itertools
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

T = 10
phi = lambda z: np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
TRIPLES = np.array([p for p in itertools.permutations(range(T), 3)])   # (720,3) positions of rank1,2,3


def forward_gates(model, x):
    """x (N,T) -> logits (N,T), gates (N,T,H)."""
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0])); G = []
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); G.append(h > 0)
    return h @ Wo.T, np.stack(G, 1)


def logit_map(model, G):
    """Exact map A (N,10 logits,10 inputs) for given gate patterns: logits = A @ x inside the cone."""
    Wi, Wh, Wo = model
    N = len(G); J = np.zeros((N, Wh.shape[0], T))
    for t in range(T):
        J = G[:, t][:, :, None] * np.einsum("ij,njt->nit", Wh, J)
        J[:, :, t] += G[:, t] * Wi[None]
    return np.einsum("ki,nit->nkt", Wo, J)


def trunc_moments(c):
    """Mean and variance of N(0,1) truncated to (-inf, c)."""
    lam = phi(c) / np.maximum(ndtr(c), 1e-300)
    return -lam, 1 - c * lam - lam ** 2


def trunc_quantiles(c, k):
    """k representative values of N(0,1) | < c (mid-quantiles), descending."""
    u = (np.arange(k) + 0.5) / k
    return norm.ppf(ndtr(c) * u)[::-1]

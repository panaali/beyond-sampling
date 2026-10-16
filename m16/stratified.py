"""Mechanism-informed stratified sampling for the accuracy of 2nd-argmax RNNs.

Insight from the M_16,10 analysis: errors concentrate where the top order statistics are nearly tied
(and in the 'blind' region where the 2nd largest is <= 0). These regions are defined by the INPUT alone,
so each stratum's probability is model-free and can be computed once, to any precision, without running
the model. Sampling within strata (model evaluations allocated where errors are) is then unbiased for any
weights, including ARC's noisy copies, and has lower variance than plain sampling at equal compute.

Note: this is still a sampling method. It uses the mechanism to decide where to sample, but it is not a
deductive 'mechanistic estimate' in ARC's sense.
"""
import numpy as np

T = 10
EDGES = np.array([0.0, 0.01, 0.025, 0.05, 0.1, 0.2, np.inf])   # bins of min relative top gap


def stratum(x):
    """0 = blind (2nd largest <= 0); 1.. = bins of min((v1-v2)/v1, (v2-v3)/v2)."""
    s = -np.sort(-x, 1)
    v1, v2, v3 = s[:, 0], s[:, 1], s[:, 2]
    blind = v2 <= 0
    g = np.minimum((v1 - v2) / np.abs(v1), (v2 - v3) / np.maximum(np.abs(v2), 1e-12))
    b = 1 + np.clip(np.searchsorted(EDGES, g, side="right") - 1, 0, len(EDGES) - 2)
    return np.where(blind, 0, b)


N_STRATA = len(EDGES)


def stratum_probs(n=20_000_000, seed=0, chunk=2_000_000):
    """Model-free probabilities of each stratum (computed once; no model evaluations)."""
    rng = np.random.default_rng(seed); counts = np.zeros(N_STRATA)
    for _ in range(n // chunk):
        counts += np.bincount(stratum(rng.standard_normal((chunk, T))), minlength=N_STRATA)
    return counts / n


def draw(h, k, rng, batch=200_000):
    """k inputs from stratum h by rejection (input generation only; no model calls)."""
    out = []
    while sum(len(o) for o in out) < k:
        x = rng.standard_normal((batch, T)); out.append(x[stratum(x) == h])
    return np.concatenate(out)[:k]


def estimate(forward, budget, rng, P, alloc):
    """Unbiased stratified estimate of accuracy. alloc: fraction of the budget per stratum."""
    n = np.maximum(2, np.round(alloc * budget).astype(int))
    acc = 0.0
    for h in range(N_STRATA):
        x = draw(h, n[h], rng)
        acc += P[h] * (forward(x).argmax(1) == np.argsort(x, 1)[:, -2]).mean()
    return acc, int(n.sum())

"""The 18 models and the per-model clock-neuron rules (fixed in the pre-registration)."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from arc_eval import Wi0, Wh0, Wo0, noisy_models


def zoo(sd):
    from alg_zoo import zoo_2nd_argmax
    m = zoo_2nd_argmax(16, 10, seed=sd); f = lambda t: t.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)


def all_models():
    """orig, zoo seeds 0-4, and noisy_models(6, s, seed=0) for s in (0.0025, 0.005).
    The first two copies of each s are exactly E27's four copies (same RNG stream)."""
    ms = {"orig": (Wi0, Wh0, Wo0)}
    for sd in range(5): ms[f"seed{sd}"] = zoo(sd)
    for s in (0.0025, 0.005):
        for i, m in enumerate(noisy_models(6, s, seed=0)): ms[f"s{s}_{i}"] = m
    return ms


def flip_scores(M, N=200_000, seed=5):
    """E27 diag_automaton2 score: per-neuron gate flips not predicted by (G_{t-1}, capped rank, sign x_t)."""
    Wi, Wh, _ = M
    x = np.random.default_rng(seed).standard_normal((N, 10)); h = np.zeros((N, 16)); B = []
    for t in range(10):
        pre = h @ Wh.T + x[:, t:t+1] * Wi[None]; B.append(pre > 0); h = np.maximum(pre, 0)
    B = np.stack(B, 1); G = B @ (1 << np.arange(16))
    rel = np.array([(x[:, :t+1] > x[:, t:t+1]).sum(1) + 1 for t in range(10)]).T
    r = np.minimum(rel, 4) * 2 + (x > 0)
    tot = np.zeros(16)
    for t in range(1, 10):
        _, inv = np.unique(G[:, t-1] * 16 + r[:, t], return_inverse=True)
        cnt = np.bincount(inv); on = np.stack([np.bincount(inv, B[:, t, j]) for j in range(16)], 1)
        tot += np.minimum(on, cnt[:, None] - on).sum(0) / N
    return tot


def clocks_data(M, m):
    return [int(j) for j in np.argsort(-flip_scores(M))[:m]]


def clocks_weight(M, m):
    """Weight-only rule: neuron participation in the complex (rotating) eigenvectors of W_hh with |lambda| >= 0.5,
    weighted by |lambda|^10 (how much of the mode survives the 10 steps)."""
    lam, V = np.linalg.eig(M[1])
    sel = (np.abs(lam.imag) > 1e-6) & (np.abs(lam) >= 0.5)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    score = (np.abs(V[:, sel]) ** 2 * (np.abs(lam[sel]) ** 10)[None]).sum(1)
    return [int(j) for j in np.argsort(-score)[:m]]


if __name__ == "__main__":
    for name, M in all_models().items():
        print(f"{name:10s} data5 {clocks_data(M, 5)}  weight5 {clocks_weight(M, 5)}", flush=True)

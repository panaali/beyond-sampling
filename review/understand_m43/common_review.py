"""Shared helpers for the T74 review. Independent of the reviewed code: own forward pass, own noise draws
(same recipe as ARC's protocol: absolute N(0, sigma^2) on all 32 weights), own MC truth."""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UND = os.path.join(ROOT, "m16", "approaches", "understand_m43")
ARC = os.path.join(ROOT, "m16", "approaches", "arc_m43_repro")
sys.path.insert(0, UND); sys.path.insert(0, ARC)


def base_model():
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(4, 3, seed=0)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"]


def zoo(seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(4, 3, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"]


def noisy(base, k, sigma, seed):
    """Same recipe as m43.noisy_models (draw order Wi, Wh, Wo per copy), re-implemented."""
    rng = np.random.default_rng(seed); Wi, Wh, Wo = base
    out = []
    for _ in range(k):
        a = Wi + sigma * rng.standard_normal(4); b = Wh + sigma * rng.standard_normal((4, 4))
        c = Wo + sigma * rng.standard_normal((3, 4)); out.append((a, b, c))
    return out


def my_forward(model, x):
    Wi, Wh, Wo = model
    h = np.maximum(np.outer(x[:, 0], Wi), 0.0)
    h = np.maximum(h @ Wh.T + np.outer(x[:, 1], Wi), 0.0)
    h = np.maximum(h @ Wh.T + np.outer(x[:, 2], Wi), 0.0)
    return h @ Wo.T


def mc_truth(model, n, seed, chunk=500_000):
    rng = np.random.default_rng(seed); ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done); x = rng.standard_normal((m, 3))
        lab = np.argsort(x, axis=1)[:, 1]
        ok += int((my_forward(model, x).argmax(1) == lab).sum()); done += m
    return ok / n

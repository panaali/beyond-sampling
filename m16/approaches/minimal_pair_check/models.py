"""Fresh models for T379: load trained checkpoints as float64 (Wi, Wh, Wo) and make the noisy copies
(pre-registered recipe: W + sigma * RMS(W) * N(0,1), sigma in SIGMAS, 2 copies each, default_rng(1000 + seed))."""
import os

import numpy as np
import torch as th

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
SEEDS = (100, 101, 102, 103, 104)
SIGMAS = (0.0025, 0.005, 0.01, 0.02)
N_COPIES = 2


def load(seed, H=16, T=5):
    sd = th.load(f"{ROOT}/data/t379_models/2nd_argmax_{H}_{T}_{2 ** 30}_{seed}.pth", weights_only=True)
    f = lambda t: t.detach().numpy().astype(np.float64)
    return f(sd["rnn.weight_ih_l0"])[:, 0], f(sd["rnn.weight_hh_l0"]), f(sd["linear.weight"])


def noisy(base, seed):
    rng = np.random.default_rng(1000 + seed); out = {}
    for sigma in SIGMAS:
        for k in range(N_COPIES):
            out[f"s{sigma}_{k}"] = tuple(W + sigma * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape)
                                         for W in base)
    return out


def all_models():
    """name -> weights; clean models are 'f<seed>', noisy copies 'f<seed>/s<sigma>_<k>'."""
    ms = {}
    for s in SEEDS:
        b = load(s); ms[f"f{s}"] = b
        for k, m in noisy(b, s).items():
            ms[f"f{s}/{k}"] = m
    return ms

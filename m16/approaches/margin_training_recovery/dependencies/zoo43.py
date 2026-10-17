"""Weights of alg_zoo.zoo_2nd_argmax(4, 3, seed=s) as (Wi (4,), Wh (4,4), Wo (3,4)) numpy arrays."""
import numpy as np


def zoo_model(seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(4, 3, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"]


def noisy_copies(base, k=256, sigma=0.05, seed=0):
    """ARC protocol on any base model (same draw order as arc_m43_repro.m43.noisy_models): N(0, sigma^2) added to
    every weight; for base = zoo seed 0 this reproduces m43.noisy_models exactly."""
    Wi, Wh, Wo = base
    rng = np.random.default_rng(seed)
    return [(Wi + sigma * rng.standard_normal(Wi.shape), Wh + sigma * rng.standard_normal(Wh.shape),
             Wo + sigma * rng.standard_normal(Wo.shape)) for _ in range(k)]

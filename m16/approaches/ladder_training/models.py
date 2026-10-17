"""Loader for the 2nd-argmax RNNs trained in T86 (ladder_training) with alg_zoo's own training code.

    from models import load, available
    Wi, Wh, Wo = load(7, 3, seed)        # numpy float64: Wi (H,), Wh (H,H), Wo (T,H); h_t = relu(Wh h + Wi x_t)

`load` returns the model trained with the setting chosen for the ladder (CHOSEN below; see README.md for why).
Pass n_tag to get another training length, e.g. load(4, 3, 0, n_tag="26s").
Each model's accuracy (2^23 N(0,1) samples, rng seed 12345) is in results.json under "ours_<H>_<T>_<n_tag>_<seed>".
For zoo sizes, use alg_zoo.zoo_2nd_argmax instead (these files exist there only as reproduction checks).
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../data/ladder_training"))

# (H, T) -> n_tag: log2(n_train), "s" = warm-up scaled to keep the zoo's schedule shape (README, Method)
CHOSEN = {(7, 3): "28s", (4, 7): "28s"}


def load(H, T, seed, n_tag=None):
    n_tag = n_tag or CHOSEN[(H, T)]
    d = np.load(os.path.join(ROOT, f"m_{H}_{T}_{n_tag}_{seed}.npz"))
    return d["Wi"].copy(), d["Wh"].copy(), d["Wo"].copy()


def accuracy(H, T, seed, n_tag=None):
    n_tag = n_tag or CHOSEN[(H, T)]
    return json.load(open(os.path.join(HERE, "results.json")))[f"ours_{H}_{T}_{n_tag}_{seed}"]["acc"]


def available():
    return sorted(os.path.basename(p)[2:-4] for p in glob.glob(os.path.join(ROOT, "m_*.npz")))

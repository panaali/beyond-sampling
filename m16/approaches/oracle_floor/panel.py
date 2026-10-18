"""DEV2 panel loader (weights from claude/t192-gpu-cost-law hp2/dev2/weights.npz, sha256 of arrays checked)."""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
DATA = os.path.join(ROOT, "data/oracle_floor")
sys.path.insert(0, os.path.join(HERE, "shim"))       # 'common' shim before m16/ (no torch / alg_zoo here)
SHA = "22d1c7c2786d5c0ac2f9c83a5d03064fe52cfaedc843f43e972ee5a66a343bec"


def dev2():
    z = np.load(os.path.join(DATA, "dev2_weights.npz"))
    assert hashlib.sha256(z["Wi"].tobytes() + z["Wh"].tobytes() + z["Wo"].tobytes()).hexdigest() == SHA
    return [(str(n), (z["Wi"][k], z["Wh"][k], z["Wo"][k])) for k, n in enumerate(z["names"])]


def target():
    import common
    return common.Wi, common.Wh, common.Wo


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def lb3():
    """LB3 test panel (m16/leaderboard.py models(), H=16, T=10): orig, seed0-4 (zoo weights from claude/t241-neuron-account
    gap_formula/zoo16_10_weights.npz, alg_zoo not installed), noisy copies s{sigma}_{k} (leaderboard._noisy, seed 0)."""
    base = target(); out = [("orig", base)]
    z = np.load(os.path.join(DATA, "zoo16_10_weights.npz"))
    out += [(f"seed{s}", (z[f"z{s}_Wi"], z[f"z{s}_Wh"], z[f"z{s}_Wo"])) for s in range(5)]
    for sigma in (0.0025, 0.005, 0.01, 0.02):
        rng = np.random.default_rng(0)
        for k in range(6):
            out.append((f"s{sigma}_{k}", tuple(W + sigma * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape) for W in base)))
    return out

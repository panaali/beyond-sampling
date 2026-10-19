"""Loader for the T421 scale-study grid (2nd-argmax RNNs, H in {16..64} x T in {10, 15, 20}).

    from models import load, load_torch, registry, panel
    Wi, Wh, Wo = load(32, 15, 3)            # numpy float64: Wi (H,), Wh (H,H), Wo (T,H); h_t = relu(Wh h + Wi x_t)
    m = load_torch(32, 15, 3)               # alg_zoo DistRNN (float32)
    panel(32, 15)                           # registry rows of the primary panel (cpu seeds 0-5)
Streams: "cpu" = alg_zoo.training.train on CPU (seed = torch seed; T392's k = 1 models are bit-identical members);
"zoo" = ARC's published zoo models (GPU-trained, seeds 0-4; only (16,10) and (32,10) exist).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def registry():
    return json.load(open(os.path.join(HERE, "registry.json")))["models"]


def load(H, T, seed, stream="cpu"):
    d = np.load(os.path.join(HERE, "weights", f"m_{H}_{T}_{stream}{seed}.npz"))
    return d["Wi"].copy(), d["Wh"].copy(), d["Wo"].copy()


def load_torch(H, T, seed, stream="cpu"):
    from alg_zoo.loading import load_dist_rnn
    return load_dist_rnn(os.path.join(HERE, "weights", f"m_{H}_{T}_{stream}{seed}.pth"))


def panel(H, T, stream="cpu"):
    return [r for r in registry() if (r["H"], r["T"], r["stream"]) == (H, T, stream)]

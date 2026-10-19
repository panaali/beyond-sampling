"""Models and truths for T393, and import of the frozen T10 estimator (sha256-asserted)."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../../..")
SHA = {"estimator.py": "fb7902e12240acd86dc6cd18786018abe73201a1996357efead379bd487fa867"}


def frozen():
    for f, h in SHA.items():
        assert hashlib.sha256(open(f"{HERE}/frozen/{f}", "rb").read()).hexdigest() == h, f
    sys.path.insert(0, f"{HERE}/frozen")
    import estimator as E  # noqa: E402
    return E


def zoo(T, seed):
    from alg_zoo import zoo_2nd_argmax
    m = zoo_2nd_argmax(16, T, seed=seed)
    f = lambda w: np.ascontiguousarray(w.detach().numpy().astype(np.float64))
    return f(m.rnn.weight_ih_l0)[:, 0].copy(), f(m.rnn.weight_hh_l0), f(m.linear.weight)


def fresh(seed):
    import torch as th
    sd = th.load(f"{ROOT}/data/t393_models/2nd_argmax_16_5_{2 ** 30}_{seed}.pth", weights_only=True)
    f = lambda t: np.ascontiguousarray(t.detach().numpy().astype(np.float64))
    return f(sd["rnn.weight_ih_l0"])[:, 0].copy(), f(sd["rnn.weight_hh_l0"]), f(sd["linear.weight"])


def groups(T):
    """name -> model for the benchmark groups at this T (fresh models only where trained)."""
    out = {f"z{s}": zoo(T, s) for s in range(5)}
    if T == 5:
        for s in range(100, 105):
            if os.path.exists(f"{ROOT}/data/t393_models/2nd_argmax_16_5_{2 ** 30}_{s}.pth"):
                out[f"f{s}"] = fresh(s)
    return out


def truths():
    p = f"{HERE}/truths.json"
    return json.load(open(p)) if os.path.exists(p) else {}

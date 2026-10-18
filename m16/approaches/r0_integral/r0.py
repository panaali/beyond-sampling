"""T297 shared code: DEV2 loader, the R0 closure, exact skeleton law. R0 (T292 oracle.py): top-3 positions/values exact,
the 7 low inputs (ascending free positions) all set to mu(v3) = -phi(v3)/Phi(v3); correct iff argmax logits = p2."""
import hashlib
import json
import os
from itertools import permutations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SHA = "22d1c7c2786d5c0ac2f9c83a5d03064fe52cfaedc843f43e972ee5a66a343bec"
T, H = 10, 16
TRIPLES = np.array(list(permutations(range(T), 3)))           # 720 x (p1, p2, p3)
DEV_IDX = list(range(0, 80, 8))


def dev2():
    z = np.load(os.path.join(HERE, "data", "dev2_weights.npz"))
    assert hashlib.sha256(z["Wi"].tobytes() + z["Wh"].tobytes() + z["Wo"].tobytes()).hexdigest() == SHA
    return [(str(n), (z["Wi"][k].astype(np.float64), z["Wh"][k].astype(np.float64), z["Wo"][k].astype(np.float64)))
            for k, n in enumerate(z["names"])]


def truths():
    return json.load(open(os.path.join(HERE, "data", "dev2_truths.json")))


def target():
    from alg_zoo import example_2nd_argmax
    m = example_2nd_argmax()
    f = lambda w: w.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)


def mu(v3):
    return -norm.pdf(v3) / np.maximum(ndtr(v3), 1e-300)


def r0_inputs(tri, v1, v2, v3):
    """tri (n,3) positions; values (n,) -> x (n,10) with free positions at mu(v3)."""
    n = len(v1); x = np.repeat(mu(v3)[:, None], T, 1)
    r = np.arange(n)
    x[r, tri[:, 0]] = v1; x[r, tri[:, 1]] = v2; x[r, tri[:, 2]] = v3
    return x


def logits(model, x, dx=None):
    """Logits and (optionally) their forward-mode tangent along dx (gates frozen; exact derivative a.e.)."""
    Wi, Wh, Wo = model
    h = np.zeros((len(x), H)); dh = np.zeros_like(h) if dx is not None else None
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        g = z > 0
        h = np.where(g, z, 0.0)
        if dx is not None:
            dh = np.where(g, dh @ Wh.T + dx[:, t:t + 1] * Wi[None], 0.0)
    return (h @ Wo.T, dh @ Wo.T) if dx is not None else h @ Wo.T


def r0_correct(model, tri, v1, v2, v3):
    return logits(model, r0_inputs(tri, v1, v2, v3)).argmax(1) == tri[:, 1]


def sample_skeleton(n, rng):
    x = rng.standard_normal((n, T)); o = np.argsort(-x, 1)[:, :3]
    v = np.take_along_axis(x, o, 1)
    return o, v[:, 0], v[:, 1], v[:, 2]


# FLOP accounting (T196: multiply-add = 2)
FLOP_FWD = T * (2 * H * H + 2 * H) + 2 * H * T                  # one forward pass incl. readout
FPE = FLOP_FWD

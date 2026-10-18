"""T340 DIAGNOSTIC (sampled inputs): for E(t0) = {pick rank 3, g23 > t0}, compare the network on x with the network on
the R1W closure point of x (v4's value at the latest free position, the other 6 lows at mu(v4); T327's R1W closure).
If the closure alone loses the far-gap events, no walk resolution can recover them. -> results/diag_closure.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "closure_floor")); sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402
from oracle import closure_point  # noqa: E402

T0 = (0.0, 1.0, 1.25, 1.5)


def logits(m, x):
    Wi, Wh, Wo = m; h = np.zeros((len(x), 16))
    for t in range(10):
        h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
    return h @ Wo.T


out = {}
ms = cf.lb3_models()
for name in ("orig", "seed0", "seed3"):
    rng = np.random.default_rng(340_900 + sum(map(ord, name))); c = np.zeros((3, len(T0))); n = 0
    for _ in range(40):
        x = rng.standard_normal((500_000, 10)); o = np.argsort(-x, 1); r = np.arange(len(x))
        g23 = x[r, o[:, 1]] - x[r, o[:, 2]]; p3 = o[:, 2]
        net = logits(ms[name], x).argmax(1) == p3
        xb, free, U, s, y = closure_point(x, "R1W"); clo = logits(ms[name], xb).argmax(1) == p3
        for i, t0 in enumerate(T0):
            far = g23 > t0
            c[0, i] += (net & far).sum(); c[1, i] += (clo & far).sum(); c[2, i] += (net & clo & far).sum()
        n += len(x)
    out[name] = {str(t0): dict(net=c[0, i] / n, closure=c[1, i] / n, both=c[2, i] / n) for i, t0 in enumerate(T0)}
    print(name, {t: f"net {v['net']:.2e} closure {v['closure']:.2e} both {v['both']:.2e}" for t, v in out[name].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "results", "diag_closure.json"), "w"), indent=1)

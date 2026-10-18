"""T304 stage A (DIAGNOSTIC sampling): oracle floors of point closures on the 80 DEV2 retrains, paired.
usage: python floors.py [nskel] -> results/floors.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
tr = r0.truths()["truths"]; out = {}


def closure_inputs(x, kind):
    o = np.argsort(-x, 1); r = np.arange(len(x)); v = np.take_along_axis(x, o, 1)
    xc = np.empty_like(x)
    if kind == "a":
        xc[:] = r0.mu(v[:, 2])[:, None]
        for k in range(3):
            xc[r, o[:, k]] = v[:, k]
        return xc
    xc[:] = r0.mu(v[:, 3])[:, None]
    for k in range(3):
        xc[r, o[:, k]] = v[:, k]
    if kind == "b":
        p4 = o[:, 3]
    else:
        free = np.ones_like(x, bool); free[r[:, None], o[:, :3]] = False
        idx = np.where(free, np.arange(10)[None], -1 if kind == "c" else 99)
        p4 = idx.max(1) if kind == "c" else idx.min(1)
    xc[r, p4] = v[:, 3]
    return xc


for k, (nm, model) in enumerate(r0.dev2()):
    rng = np.random.default_rng(304_000 + k); acc = {c: 0 for c in "abcd"}
    for s in range(0, n, 100_000):
        x = rng.standard_normal((100_000, 10)); y = np.argsort(-x, 1)[:, 1]
        for c in "abcd":
            acc[c] += (r0.logits(model, closure_inputs(x, c)).argmax(1) == y).sum()
    out[nm] = {c: float(acc[c] / n - tr[nm]["acc"]) for c in "abcd"}
    print(nm, {c: round(100 * v, 2) for c, v in out[nm].items()}, flush=True)
summ = {c: float(100 * np.sqrt(np.mean([out[m][c] ** 2 for m in out]))) for c in "abcd"}
summ_mean = {c: float(100 * np.mean([out[m][c] for m in out])) for c in "abcd"}
print("RMS floor pp", {c: round(v, 3) for c, v in summ.items()}, "mean", {c: round(v, 3) for c, v in summ_mean.items()})
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(dict(per_model=out, rms_pp=summ, mean_pp=summ_mean, n=n), open(os.path.join(HERE, "results", "floors.json"), "w"), indent=1)

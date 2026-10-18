"""T274b, DRI-suggested deductive baseline (DIAGNOSTIC of its approximation bias; random inputs only to measure it):
branch_bound's 180 root cones (a = argmax, b = 2nd, s = sign of x_b) with exact masses, but the cone's verdict
derived from the weights: gates frozen at a weight-free canonical input of the cone (model-free conditional means
of the max / 2nd / remaining values given s, placed at a / b / the rest), so logits are an exact linear map on the
cone, integrated exactly over the cone's Gaussian law (model-free). Bias of that approximation vs the network,
paired, for m = 0..8 exact last steps.  usage: python cone_bias.py [n] -> results/cone_bias.json"""
import json
import math
import os
import sys

import numpy as np

from frozen_bias import HERE, H, T, approx_pred, gates, lb3_models

MS = (0, 1, 2, 4, 6, 8)
_r = np.sort(np.random.default_rng(1).standard_normal((2_000_000, T)), 1)[:, ::-1]   # model-free table
S = _r[:, 1] > 0
CM = {s: (_r[S == s, 0].mean(), _r[S == s, 1].mean(), _r[S == s, 2:].mean()) for s in (True, False)}
del _r, S


def canonical_cone(x):
    o = np.argsort(-x, 1); a, b = o[:, 0], o[:, 1]; s = x[np.arange(len(x)), b] > 0
    xc = np.empty_like(x); rows = np.arange(len(x))
    for sv in (True, False):
        m = s == sv
        xc[m] = CM[sv][2]; xc[rows[m], a[m]] = CM[sv][0]; xc[rows[m], b[m]] = CM[sv][1]
    return xc


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    out = {}
    for name, model in lb3_models().items():
        rng = np.random.default_rng(275_000 + sum(map(ord, name))); acc = {}
        Wi, Wh, Wo = model
        for s in range(0, n, 100_000):
            x = rng.standard_normal((min(100_000, n - s), T)); y = np.argsort(-x, 1)[:, 1]
            h = np.zeros((len(x), H))
            for t in range(T):
                h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
            ok = (h @ Wo.T).argmax(1) == y
            G = gates(model, canonical_cone(x))
            for m in MS:
                acc.setdefault(f"cone_m{m}", []).append(((approx_pred(model, x, G, m) == y).astype(int) - ok).sum())
            acc.setdefault("net", []).append(ok.sum())
        out[name] = {k: float(np.sum(v) / n) for k, v in acc.items()}
        print(name, {k: round(100 * v, 3) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "cone_bias.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

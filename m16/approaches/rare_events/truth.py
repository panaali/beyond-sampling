"""T340 truth: plain MC, N inputs per model (chunks of 500k), counts of E(t0) = {network picks the rank-3 position and
g23 > t0} for t0 in T0. One process. usage: python truth.py [N] -> results/truth.json (updated after every model)"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402

T0 = (0.0, 1.0, 1.25, 1.5)
MODELS = ("orig", "seed0", "seed3")


NPER = {"orig": 1_000_000_000, "seed0": 300_000_000, "seed3": 300_000_000}   # declared deviation (CPU budget)


def main(N=None, chunk=500_000):
    ms = cf.lb3_models(); path = os.path.join(HERE, "results", "truth.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in MODELS:
        Nm = N or NPER[name]
        if name in out and out[name]["n"] >= Nm:
            continue
        Wi, Wh, Wo = ms[name]; WhT = Wh.T.copy(); WoT = Wo.T.copy()
        rng = np.random.default_rng(340_000 + sum(map(ord, name))); cnt = np.zeros(len(T0), np.int64); n = 0; t0 = time.time()
        while n < Nm:
            x = rng.standard_normal((chunk, 10)); h = np.zeros((chunk, 16))
            for t in range(10):
                h = h @ WhT; h += x[:, t:t + 1] * Wi; np.maximum(h, 0, out=h)
            yhat = (h @ WoT).argmax(1)
            idx = np.argpartition(-x, 2, axis=1)[:, :3]                       # top-3 positions (unordered)
            vals = np.take_along_axis(x, idx, 1); o = np.argsort(-vals, 1)
            p3 = np.take_along_axis(idx, o[:, 2:3], 1)[:, 0]
            v = np.take_along_axis(vals, o, 1); g23 = v[:, 1] - v[:, 2]
            hit = yhat == p3
            for i, t0_ in enumerate(T0):
                cnt[i] += int((hit & (g23 > t0_)).sum())
            n += chunk
        out[name] = dict(n=n, counts={str(t): int(c) for t, c in zip(T0, cnt)},
                         p={str(t): c / n for t, c in zip(T0, cnt)}, sec=time.time() - t0)
        json.dump(out, open(path, "w"), indent=1)
        print(name, out[name], flush=True)


if __name__ == "__main__":
    main(int(float(sys.argv[1])) if len(sys.argv) > 1 else None)

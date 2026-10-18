"""T340 E2 truth (CPU): counts of E2(t) = {pick != p2 and g12 >= t and g23 >= t}, t in T2, plus all errors.
usage: python truth2.py [N per model] -> results/truth2.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402

T2 = (0.5, 0.75)


def main(N=200_000_000, chunk=500_000):
    ms = cf.lb3_models(); path = os.path.join(HERE, "results", "truth2.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in ("orig", "seed0", "seed3"):
        if name in out and out[name]["n"] >= N:
            continue
        Wi, Wh, Wo = ms[name]; WhT = Wh.T.copy(); WoT = Wo.T.copy()
        rng = np.random.default_rng(340_500 + sum(map(ord, name))); cnt = np.zeros(len(T2) + 1, np.int64); n = 0; t0 = time.time()
        while n < N:
            x = rng.standard_normal((chunk, 10)); h = np.zeros((chunk, 16))
            for t in range(10):
                h = h @ WhT; h += x[:, t:t + 1] * Wi; np.maximum(h, 0, out=h)
            yhat = (h @ WoT).argmax(1)
            idx = np.argpartition(-x, 2, axis=1)[:, :3]
            vals = np.take_along_axis(x, idx, 1); o = np.argsort(-vals, 1)
            p2 = np.take_along_axis(idx, o[:, 1:2], 1)[:, 0]
            v = np.take_along_axis(vals, o, 1); g12 = v[:, 0] - v[:, 1]; g23 = v[:, 1] - v[:, 2]
            wrong = yhat != p2
            for i, t in enumerate(T2):
                cnt[i] += int((wrong & (g12 >= t) & (g23 >= t)).sum())
            cnt[-1] += int(wrong.sum()); n += chunk
        out[name] = dict(n=n, counts={**{str(t): int(c) for t, c in zip(T2, cnt)}, "all_errors": int(cnt[-1])},
                         p={**{str(t): c / n for t, c in zip(T2, cnt)}, "all_errors": cnt[-1] / n}, sec=time.time() - t0)
        json.dump(out, open(path, "w"), indent=1); print(name, out[name], flush=True)


if __name__ == "__main__":
    main(int(float(sys.argv[1])) if len(sys.argv) > 1 else 200_000_000)

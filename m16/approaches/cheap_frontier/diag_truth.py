"""DIAGNOSTIC (sampling; for error attribution only, never part of an estimator): per LB3 model, the true error
split by (a) number K of positive inputs and (b) which rank the network picks, separately for the blind region
(2nd largest <= 0) and b > 0. n samples per model (default 2M). -> results/truth_parts.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402


def main(n=2_000_000, chunk=250_000):
    out = {}
    for name, (Wi, Wh, Wo) in cf.lb3_models().items():
        rng = np.random.default_rng(286_000 + sum(map(ord, name)))
        errK = np.zeros(11); pick = np.zeros((2, 10)); cnt = 0
        for _ in range(n // chunk):
            x = rng.standard_normal((chunk, 10)); h = np.zeros((chunk, 16))
            for t in range(10):
                h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
            y = (h @ Wo.T).argmax(1); order = np.argsort(-x, 1)
            rank = np.argmax(order == y[:, None], 1)          # rank (0 = max) of the picked position
            K = (x > 0).sum(1); blind = x[np.arange(chunk), order[:, 1]] <= 0
            wrong = rank != 1
            errK += np.bincount(K[wrong], minlength=11)
            for bl in (0, 1):
                s = (blind == bl) & wrong
                pick[bl] += np.bincount(rank[s], minlength=10)
            cnt += chunk
        out[name] = dict(n=cnt, errK=(errK / cnt).tolist(), pick_pos=(pick[0] / cnt).tolist(),
                         pick_blind=(pick[1] / cnt).tolist(), err=float(errK.sum() / cnt))
        print(name, round(out[name]["err"], 5), flush=True)
    json.dump(out, open(os.path.join(cf.HERE, "results", "truth_parts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

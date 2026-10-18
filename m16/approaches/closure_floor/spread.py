"""T327 arm B (addendum): one-pattern spread lows. Hard verdict at the closure point with each free low at absolute
position j set to mu + sgn * (-1)^j sigma (mu, sigma of N(0,1) | < U); sgn = +1 (pre-registered) and -1 (check).
Paired with the network on the same inputs. DIAGNOSTIC (sampled inputs).
usage: python spread.py [n] -> results/spread.jsonl"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle import CLOSURES, H, T, closure_point, cf  # noqa: E402

SIGN = (-1.0) ** np.arange(T)


def logits(model, x):
    Wi, Wh, Wo = model; h = np.zeros((len(x), H))
    for t in range(T):
        h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
    return h @ Wo.T


def run_model(model, name, n, chunk=50_000):
    rng = np.random.default_rng(327_500 + sum(map(ord, name))); acc = {}; cnt = 0
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T)); y = np.argsort(-x, 1)[:, 1]
        rows = {"net": logits(model, x).argmax(1) == y}
        for kind in CLOSURES:
            xb, free, U, s, yy = closure_point(x, kind)
            rows[f"{kind}:mean"] = logits(model, xb).argmax(1) == yy
            for sg, lab in ((1.0, "spread+"), (-1.0, "spread-")):
                xs = np.where(free, xb + sg * SIGN[None] * s[:, None], xb)
                rows[f"{kind}:{lab}"] = logits(model, xs).argmax(1) == yy
        for k, v in rows.items():
            acc[k] = acc.get(k, 0.0) + float(v.sum())
        cnt += chunk
    return dict(model=name, n=cnt, mean={k: v / cnt for k, v in acc.items()})


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    ms = cf.dev2_models(); tr = cf.dev2_truths()
    path = os.path.join(HERE, "results", "spread.jsonl")
    with open(path, "w") as f:
        rows = []
        for name, m in ms.items():
            r = run_model(m, name, n); rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
    print(f"{'closure:pattern':16s} {'RMS pp':>7s} {'mean pp':>8s}")
    for k in rows[0]["mean"]:
        if k == "net":
            continue
        e = np.array([r["mean"][k] - r["mean"]["net"] + r["mean"]["net"] - tr[r["model"]]["acc"] for r in rows])
        print(f"{k:16s} {100 * np.sqrt((e ** 2).mean()):7.3f} {100 * e.mean():+8.3f}")


if __name__ == "__main__":
    main()

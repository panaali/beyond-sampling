"""PROVISIONAL DEV2 truths (plain MC, n per model, numpy, own seeds) until T284 publishes its 1e8-sample truths.
-> ext/dev2/truths_provisional.json  {name: {acc, se, n}}"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402


def main(n=4_000_000, chunk=250_000):
    out = {}
    for name, (Wi, Wh, Wo) in cf.dev2_models().items():
        rng = np.random.default_rng(286_500 + sum(map(ord, name))); ok = 0
        for _ in range(n // chunk):
            x = rng.standard_normal((chunk, 10)); h = np.zeros((chunk, 16))
            for t in range(10):
                h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
            ok += int(((h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
        p = ok / n; out[name] = dict(acc=p, se=float(np.sqrt(p * (1 - p) / n)), n=n)
        print(name, round(p, 5), flush=True)
    json.dump(out, open(os.path.join(cf.HERE, "ext", "dev2", "truths_provisional.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

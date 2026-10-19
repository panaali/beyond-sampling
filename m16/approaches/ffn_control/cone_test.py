"""DRI idea check for depth-1 nets: are first-layer 'comparators' exact (gate == [x_a > x_b]), is the net close to
permutation-equivariant, and how many gate subregions does one ordering cone contain?

usage: python cone_test.py D1w128_s0_n28 [...]    -> results/cone_test.json (sampled; all counts are lower bounds)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def check(m, seed=0, n=200_000, n_perm=20, n_cones=50, per_cone=20_000):
    W1 = m.W[0]
    U = W1 / np.linalg.norm(W1, axis=1)[:, None]
    srt = np.sort(np.abs(U), 1)[:, ::-1]
    pair = srt[:, 1] > .6                       # two dominant entries of opposite sign: relu(~ x_a - x_b)
    A, B = np.argmax(U, 1), np.argmin(U, 1)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 10))
    g = x @ W1.T > 0
    mism = g[:, pair] != (x[:, A] > x[:, B])[:, pair]
    pred = m.forward(x).argmax(1)
    agree = []
    for _ in range(n_perm):
        pi = rng.permutation(10)
        agree.append(float((m.forward(x[:, pi]).argmax(1) == np.argsort(pi)[pred]).mean()))
    sub = []
    for _ in range(n_cones):
        pi = rng.permutation(10)
        s = -np.sort(-rng.standard_normal((per_cone, 10)), 1)
        xc = np.empty_like(s); xc[:, pi] = s
        sub.append(len({k.tobytes() for k in np.packbits(xc @ W1.T > 0, axis=1)}))
    return {"comparators": int(pair.sum()), "width": int(W1.shape[0]),
            "comparator_gate_mismatch_rate": float(mism.mean()), "inputs_with_any_mismatch": float(mism.any(1).mean()),
            "perm_decision_agreement": float(np.mean(agree)),
            "own_accuracy": float((pred == np.argsort(x, 1)[:, -2]).mean()),
            "gate_patterns_per_cone_median": int(np.median(sub)), "gate_patterns_per_cone_max": int(max(sub)),
            "per_cone_samples": per_cone}


if __name__ == "__main__":
    res = {name: check(analyze.load(f"{HERE}/models/{name}.json")) for name in sys.argv[1:]}
    for k, v in res.items():
        print(k, v)
    json.dump(res, open(f"{HERE}/results/cone_test.json", "w"), indent=1)

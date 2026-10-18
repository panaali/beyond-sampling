"""T369 step 2: closure oracle for E1(t0) = {pick = p3, g23 > t0} on dev3_s547 (README, T369 pre-registration).
Exact sampling given g23 > t0, stratified in g12 (A: [0, .1], B: > .1) with exact per-sample stratum weights; worlds
W0 (the estimator's closure), W2 (exact v4 law, lows at mu(v4)), W3 (exact lows, v4 at the latest free position), W4
(truth: random positions), as T363. Oracle only. usage: python e1_oracle.py -> results_e1_oracle.json"""
import json
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
import e2_oracle as O  # noqa: E402

N, CH = 2_000_000, 250_000
CUT = 0.1


def sample(t0, stratum, n, rng):
    g = np.linspace(-8, 8, 160001); w = 90 * O.pdf(g) * ndtr(-g) * ndtr(g - t0) ** 8
    c = np.cumsum(w); c /= c[-1]; v2 = np.interp(rng.random(n), c, g)
    lo, hi = (ndtr(v2), ndtr(v2 + CUT)) if stratum == "A" else (ndtr(v2 + CUT), np.ones(n))
    sw = (hi - lo) / ndtr(-v2)                                             # P(stratum | v2)
    u = lo + rng.random(n) * (hi - lo); v1 = np.where(stratum == "A", ndtri(u), -ndtri(np.maximum(1 - u, 1e-300)))
    v3 = ndtri(ndtr(v2 - t0) * rng.random(n) ** (1 / 8)); lows7 = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))
    return v1, v2, v3, lows7, rng.integers(len(E.TRI), size=n), sw


def hit(model, x, tri):
    Wi, Wh, Wo = model; h = np.zeros((len(x), 16))
    for s in range(10):
        h = np.maximum(h @ Wh.T + np.outer(x[:, s], Wi), 0)
    return ((h @ Wo.T).argmax(1) == E.TRI[tri, 2]).astype(float)


if __name__ == "__main__":
    z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
    nm = "dev3_s547"; i = names.index(nm); model = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
    truth = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "t363", "t367_truth_gpu.json"))); out = {}
    for t0 in (2.25, 2.5):
        Pg = quad(lambda v: 90 * O.pdf(v) * ndtr(-v) * ndtr(v - t0) ** 8, -12, 12, epsrel=1e-12, limit=400)[0]
        p_true = truth[f"{nm}@{t0:g}"]["p"]; rec = dict(truth=p_true, P_g23=Pg)
        for st in ("A", "B"):
            acc = {w: 0.0 for w in ("W0", "W2", "W3", "W4")}; rng = np.random.default_rng(369000 + int(t0 * 4) * 10 + (st == "B"))
            for s in range(0, N, CH):
                v1, v2, v3, lows7, tri, sw = sample(t0, st, CH, rng)
                for w in acc:
                    acc[w] += (sw * hit(model, O.build(w, v1, v2, v3, lows7, tri, rng), tri)).sum() / N
            rec[st] = {w: Pg * v for w, v in acc.items()}
        rec["ratio_to_truth"] = {w: (rec["A"][w] + rec["B"][w]) / p_true for w in ("W0", "W2", "W3", "W4")}
        out[str(t0)] = rec; print(t0, json.dumps(rec), flush=True)
    json.dump(out, open(os.path.join(HERE, "results_e1_oracle.json"), "w"), indent=1)

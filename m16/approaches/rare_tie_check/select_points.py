"""T403 step 3: truth-blind selection of (model, t0) points from the sealed estimates only (README pre-registration).
-> results/selection.json (points, predicted share s, truth sample size n0)."""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from truth import Z  # noqa: E402  (model-free normaliser P(g23 > t0) only)

NMAX_FEAS = 6e8          # feasibility: 1000 expected hits must need <= this many conditional samples
PER_STRATUM = 10
A, B = (0.5, 0.8), (0.8, 1.01)


def load(pattern):
    est = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results", pattern))):
        for nm, r in json.load(open(f)).items():
            for t, v in r["est"].items():
                est[(nm, float(t))] = dict(est=v, s=r["tie"][t] / v if v > 0 else None, C=r["C_fpe"])
    return est


def pick(cands, key):
    """One candidate per model (best by key), then up to PER_STRATUM spread evenly over sorted s."""
    best = {}
    for c in cands:
        if c["model"] not in best or key(c) < key(best[c["model"]]):
            best[c["model"]] = c
    xs = sorted(best.values(), key=lambda c: (c["s"], c["model"]))
    if len(xs) > PER_STRATUM:
        xs = [xs[i] for i in sorted(set(int(round(j)) for j in np.linspace(0, len(xs) - 1, PER_STRATUM)))]
    return xs


def select(est):
    zs = {}; cands = []
    for (nm, t), r in est.items():
        if r["s"] is None or r["est"] <= 0:
            continue
        zs.setdefault(t, Z(t)); nreq = 1000 * zs[t] / r["est"]
        if nreq <= NMAX_FEAS:
            cands.append(dict(model=nm, t0=t, est=r["est"], s=r["s"], C=r["C"], n0=int(min(1e9, max(2e7, math.ceil(2 * nreq / 1e7) * 1e7)))))
    a = pick([c for c in cands if A[0] <= c["s"] < A[1]], key=lambda c: (abs(c["s"] - 0.65), c["t0"]))
    b = pick([c for c in cands if c["s"] >= B[0]], key=lambda c: (-c["s"], c["t0"]))
    for c in a:
        c["stratum"] = "A 50-80%"
    for c in b:
        c["stratum"] = "B >=80%"
    return a, b, len(cands)


if __name__ == "__main__":
    est = load("est_[0-9]*.json"); a, b, nc = select(est); ext = False
    if min(len(a), len(b)) < 6:                    # pre-registered fallback: extended grid (est_ext_*.json)
        e2 = load("est_ext_*.json")
        if e2:
            est.update(e2); a, b, nc = select(est); ext = True
    out = dict(n_points_scanned=len(est), n_feasible=nc, extended_grid=ext, A=a, B=b)
    json.dump(out, open(os.path.join(HERE, "results", "selection.json"), "w"), indent=1)
    for c in a + b:
        print(c["stratum"], c["model"], c["t0"], "est %.2e s %.3f n0 %.0e" % (c["est"], c["s"], c["n0"]))
    print("A", len(a), "B", len(b), "feasible", nc, "ext", ext)

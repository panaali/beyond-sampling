"""Score exact-walk estimators against plain sampling on a panel: F = mean_m(Var_m c_m) / mean_m p_m(1-p_m).

For an unbiased estimator that averages n iid walks of per-walk variance Var_m and mean cost c_m (FPE), at budget
C (>= 1k FPE) n = C / c_m and MSE_m = Var_m c_m / C, so F = MSE C / p(1-p) does not depend on C. Plain sampling
has Var = p(1-p), c = 1, F = 1. We also report F_floor, which charges each piece only the floor cost of any walk
algorithm (one logit-vector update with slope, 4T FLOPs) plus envelope/erf work and 2 FPE per walk, and F_incr,
which charges each gate flip only the work after the flipped layer/step (walks.py model.suffix).

usage: python score.py ARCH_PREFIX N_COORD N_CIRCLE      e.g. python score.py M16_10 3000 600
writes results/walks_<ARCH_PREFIX>.json (per-model values) and prints F with a bootstrap CI over models.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
import walks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(prefix, n_coord, n_circle):
    models = {k: v for k, v in analyze.panel().items()
              if k == prefix or k.startswith(prefix + "_s") or k.startswith(prefix + "_n")}
    out = {}
    for name, m in models.items():
        t0 = time.time()
        p = analyze.accuracy(m, 10_000_000, 4242)
        rec = {"p_mc_1e7": p, "fwd_flops": m.fwd_flops}
        for kind, fn, n in (("coord", walks.coord_walk, n_coord), ("circle", walks.circle_walk, n_circle)):
            rng = np.random.default_rng([345, len(out), n])
            v = np.array([fn(m, rng) for _ in range(n)])
            floor = 2 * m.fwd_flops + v[:, 2] * 4 * walks.T + v[:, 3]
            rec[kind] = {"n": n, "mean": float(v[:, 0].mean()), "var": float(v[:, 0].var(ddof=1)),
                         "cost_fpe": float(v[:, 1].mean() / m.fwd_flops), "pieces": float(v[:, 2].mean()),
                         "cost_floor_fpe": float(floor.mean() / m.fwd_flops),
                         "cost_incr_fpe": float(v[:, 4].mean() / m.fwd_flops),
                         "z_vs_mc": float((v[:, 0].mean() - p) / np.sqrt(v[:, 0].var(ddof=1) / n + p * (1 - p) / 1e7))}
        out[name] = rec
        print(name, f"{time.time() - t0:.0f}s", json.dumps(rec), flush=True)
    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump(out, open(f"{HERE}/results/walks_{prefix}.json", "w"), indent=1)
    return out


def figure(out, kind, cost_key="cost_fpe", n_boot=10000, seed=0):
    names = list(out)
    num = np.array([out[k][kind]["var"] * out[k][kind][cost_key] for k in names])
    den = np.array([out[k]["p_mc_1e7"] * (1 - out[k]["p_mc_1e7"]) for k in names])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(names), size=(n_boot, len(names)))
    boot = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]


if __name__ == "__main__":
    prefix = sys.argv[1]
    out = run(prefix, int(sys.argv[2]), int(sys.argv[3]))
    for kind in ("coord", "circle"):
        print(prefix, kind, "F", figure(out, kind), "F_incr", figure(out, kind, "cost_incr_fpe"),
              "F_floor", figure(out, kind, "cost_floor_fpe"))

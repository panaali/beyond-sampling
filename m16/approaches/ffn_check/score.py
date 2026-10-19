"""T394 pipeline (pre-registered in README). Stages, in order:
  python score.py acc        accuracy on 1e7 inputs (seed 39401) + comparator counts  -> results/models.json
  python score.py n          circles per fixed set from 200 random circles on s101     -> results/n.json
  python score.py validate   brute-force check (20 circles per model, 1e6-point grid)  -> results/validate.json
  python score.py truth      1e8-sample truths (seed 39402)                            -> results/truth.json
  python score.py random     2e4 random circles per model (seed 39403), F_rand         -> results/random.json
  python score.py fixed      Halton seeds 11-15 (primary) and 1-5 (secondary)          -> results/fixed.json
"""
import json
import os
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import walk

SEEDS = [101, 102, 103, 104, 105, 106]
HERE = os.path.dirname(os.path.abspath(__file__))
R = f"{HERE}/results"


def models():
    return {s: walk.load(f"{HERE}/models/s{s}.json") for s in SEEDS}


def save(name, obj):
    json.dump(obj, open(f"{R}/{name}.json", "w"), indent=1)


def comparators(W1):
    U = W1 / np.linalg.norm(W1, axis=1, keepdims=True)
    mine = (U.max(1) > .6) & (U.min(1) < -.6)
    pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(U.argmax(1)[mine], U.argmin(1)[mine])}
    t345 = np.sort(np.abs(U), 1)[:, -2] > .6
    ideal = np.zeros_like(U); ideal[np.arange(len(U)), U.argmax(1)] = 2 ** -.5; ideal[np.arange(len(U)), U.argmin(1)] = -2 ** -.5
    resid = np.linalg.norm(U - ideal, axis=1)[mine]
    return {"comparators": int(mine.sum()), "pairs_covered": len(pairs), "t345_rule": int(t345.sum()),
            "resid_median": float(np.median(resid)), "resid_max": float(resid.max()),
            "row_sum_over_norm_max": float(np.abs(W1.sum(1) / np.linalg.norm(W1, axis=1)).max())}


def walks(W1, W2, UV):
    r = [walk.circle(W1, W2, u, v) for u, v in UV]
    return np.array([a for a, _ in r]), np.array([f for _, f in r]) / walk.fwd_flops(W1, W2)


def rand_uv(seed, n):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(10), rng.standard_normal(10)) for _ in range(n)]


def main(stage):
    M = models()
    if stage == "acc":
        out = {}
        for s, (W1, W2) in M.items():
            p = walk.accuracy(W1, W2, 10_000_000, 39401)
            out[s] = {"acc": p, "se": (p * (1 - p) / 1e7) ** .5, **comparators(W1)}
            print(s, out[s], flush=True)
        save("models", out)
    elif stage == "n":
        _, c = walks(*M[101], rand_uv(39404, 200))
        save("n", {"cost_fpe_per_circle": float(c.mean()), "n": int(round(4000 / c.mean()))}); print(c.mean(), round(4000 / c.mean()))
    elif stage == "validate":
        out = {}
        for s, (W1, W2) in M.items():
            d = [walk.circle(W1, W2, u, v)[0] - walk.brute(W1, W2, u, v, 1_000_000) for u, v in rand_uv(39405 + s, 20)]
            out[s] = float(np.abs(d).max()); print(s, out[s], flush=True)
        save("validate", out)
    elif stage == "truth":
        out = {}
        for s, (W1, W2) in M.items():
            out[s] = walk.accuracy(W1, W2, 100_000_000, 39402); print(s, out[s], flush=True)
            save("truth", out)
    elif stage == "random":
        truth = json.load(open(f"{R}/truth.json"))
        out = {}
        for s, (W1, W2) in M.items():
            e, c = walks(W1, W2, rand_uv(39403, 20_000))
            p = truth[str(s)]
            out[s] = {"mean": float(e.mean()), "z": float((e.mean() - p) / (e.std() / len(e) ** .5)), "var": float(e.var()),
                      "cost": float(c.mean()), "pq": p * (1 - p)}
            print(s, out[s], flush=True)
        out["F_rand"] = float(np.mean([o["var"] * o["cost"] for o in out.values()]) / np.mean([o["pq"] for o in out.values()]))
        print("F_rand", out["F_rand"]); save("random", out)
    elif stage == "fixed":
        truth = json.load(open(f"{R}/truth.json")); n = json.load(open(f"{R}/n.json"))["n"]
        pq = np.array([truth[str(s)] * (1 - truth[str(s)]) for s in SEEDS])
        out = {"n": n, "sets": {}}
        for hs in [11, 12, 13, 14, 15, 1, 2, 3, 4, 5]:
            q = qmc.Halton(d=20, scramble=True, seed=hs).random(n)
            UV = list(zip(ndtri(q[:, :10]), ndtri(q[:, 10:])))
            err, ceff = [], []
            for s, (W1, W2) in M.items():
                e, c = walks(W1, W2, UV)
                err.append(e.mean() - truth[str(s)]); ceff.append(max(c.sum(), 1000.0))
            err, ceff = np.array(err), np.array(ceff)
            out["sets"][hs] = {"F": float((err ** 2 * ceff).mean() / pq.mean()), "rms_err": float(np.sqrt((err ** 2).mean())),
                               "mean_err": float(err.mean()), "C_exec": float(ceff.mean()), "errs": err.tolist()}
            print(hs, out["sets"][hs], flush=True)
        for fam, ss in (("primary", [11, 12, 13, 14, 15]), ("secondary", [1, 2, 3, 4, 5])):
            F = [out["sets"][k]["F"] for k in ss]
            out[fam] = {"F": F, "mean": float(np.mean(F)), "min": min(F), "max": max(F), "all_below_1": bool(max(F) < 1)}
            print(fam, out[fam])
        save("fixed", out)


if __name__ == "__main__":
    main(sys.argv[1])

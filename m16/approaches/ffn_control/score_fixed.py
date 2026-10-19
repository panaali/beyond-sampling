"""T387: fixed (deterministic) great-circle sets for depth-1 nets, scored at C = 4,000 FPE (pre-registered in README).

Every model gets the same circles; no random model inputs. F = mean_m e_m^2 * C_eff / mean_m p_m(1-p_m), with
e_m = estimate - truth (1e8-sample MC, seed 8787, cached in results/truth1e8_<ARCH>.json).

usage: python score_fixed.py D1w128      -> results/fixed_D1w128.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
import fast1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, C = 232, 4000


def designs():
    out = {}
    for s in range(1, 6):
        out[f"S{s}"] = qmc.Sobol(d=20, scramble=True, seed=s).random(N)
        out[f"H{s}"] = qmc.Halton(d=20, scramble=True, seed=s).random(N)
        base = qmc.Sobol(d=20, scramble=True, seed=s).random(23)
        P = []
        for q in base:
            for k in range(10):
                P.append(np.concatenate([np.roll(q[:10], k), np.roll(q[10:], k)]))
        out[f"P{s}"] = np.array(P)
    n = 233
    for a in (17, 31, 57, 89, 113):
        z = np.array([pow(a, j, n) for j in range(20)])
        out[f"K{a}"] = np.mod((np.arange(n)[:, None] * z[None] + 0.5) / n, 1.0)
    return out


def truths(arch, models):
    path = f"{HERE}/results/truth1e8_{arch}.json"
    t = json.load(open(path)) if os.path.exists(path) else {}
    for name, m in models.items():
        if name not in t:
            t[name] = analyze.accuracy(m, 100_000_000, 8787)
            json.dump(t, open(path, "w"), indent=1)
            print("truth", name, t[name], flush=True)
    return t


def main(arch):
    models = analyze.panel([arch])
    truth = truths(arch, models)
    pq = np.array([truth[k] * (1 - truth[k]) for k in models])
    res = {"N": N, "C_target": C, "truth_se": float(np.sqrt(pq.mean() / 1e8)), "sets": {}}
    rng = np.random.default_rng(0)
    for dname, Q in designs().items():
        t0 = time.time()
        U, V = ndtri(Q[:, :10]), ndtri(Q[:, 10:])
        est, cost = {}, {}
        for name, m in models.items():
            r = [fast1.circle(m.W[0], m.W[1], u, v) for u, v in zip(U, V)]
            est[name] = float(np.mean([a for a, _ in r]))
            cost[name] = float(sum(f for _, f in r) / m.fwd_flops)
        e2 = np.array([(est[k] - truth[k]) ** 2 for k in models])
        ceff = np.array([max(cost[k], 1000.0) for k in models])
        num = e2 * ceff
        idx = rng.integers(len(num), size=(10000, len(num)))
        b = num[idx].mean(1) / pq[idx].mean(1)
        res["sets"][dname] = {"F": float(num.mean() / pq.mean()), "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                              "rms_err": float(np.sqrt(e2.mean())), "mean_err": float(np.mean([est[k] - truth[k] for k in models])),
                              "cost_fpe": float(ceff.mean()), "n_circles": len(Q)}
        print(dname, f"{time.time() - t0:.0f}s", res["sets"][dname], flush=True)
    for fam in "SHPK":
        Fs = [v["F"] for k, v in res["sets"].items() if k[0] == fam]
        res[f"family_{fam}"] = {"F_mean": float(np.mean(Fs)), "F_min": float(min(Fs)), "F_max": float(max(Fs))}
        print("family", fam, res[f"family_{fam}"])
    json.dump(res, open(f"{HERE}/results/fixed_{arch}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

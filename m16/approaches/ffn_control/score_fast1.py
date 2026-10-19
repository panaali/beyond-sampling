"""Score the fast exact great-circle walk (fast1.py) on a depth-1 panel; F = mean_m(Var_m c_m) / mean_m p_m(1-p_m).

Also a direct budget check: at C = 4000 FPE, REPS estimates per model, each the mean of floor(C / c_m) fresh walks,
compared with an independent 1e7-sample truth; F_direct = mean MSE * C / mean p(1-p) (includes the truth's noise).

usage: python score_fast1.py ARCH N_WALKS REPS        e.g. python score_fast1.py D1w128 3000 40
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
import fast1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def boot(num, den, n_boot=10000, seed=0):
    idx = np.random.default_rng(seed).integers(len(num), size=(n_boot, len(num)))
    b = num[idx].mean(1) / den[idx].mean(1)
    return [float(num.mean() / den.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main(arch, n_walks, reps, C=4000):
    out = {}
    for k, (name, m) in enumerate(analyze.panel([arch]).items()):
        t0 = time.time()
        p = analyze.accuracy(m, 10_000_000, 4242)
        truth = analyze.accuracy(m, 10_000_000, 4343)
        rng = np.random.default_rng([345, 1, k])
        v = np.array([fast1.walk(m, rng)[:2] for _ in range(n_walks)])
        var, c = float(v[:, 0].var(ddof=1)), float(v[:, 1].mean() / m.fwd_flops)
        nb = int(C // c)
        est = [np.mean([fast1.walk(m, rng)[0] for _ in range(nb)]) for _ in range(reps)]
        out[name] = {"p_mc_1e7": p, "truth_1e7": truth, "var": var, "cost_fpe": c, "mean": float(v[:, 0].mean()),
                     "z_vs_mc": float((v[:, 0].mean() - p) / np.sqrt(var / n_walks + p * (1 - p) / 1e7)),
                     "direct": {"C": C, "walks": nb, "reps": reps, "mse": float(np.mean((np.array(est) - truth) ** 2))}}
        print(name, f"{time.time() - t0:.0f}s", json.dumps(out[name]), flush=True)
    names = list(out)
    pq = np.array([out[k]["p_mc_1e7"] * (1 - out[k]["p_mc_1e7"]) for k in names])
    res = {"models": out,
           "F": boot(np.array([out[k]["var"] * out[k]["cost_fpe"] for k in names]), pq),
           "F_direct_C4000": boot(np.array([out[k]["direct"]["mse"] * C for k in names]), pq)}
    json.dump(res, open(f"{HERE}/results/fast1_{arch}.json", "w"), indent=1)
    print(arch, "F", res["F"], "F_direct", res["F_direct_C4000"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

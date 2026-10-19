"""T379 scoring: T10's frozen estimator (sha256-asserted) on the fresh models, against own truths (truths.json).
usage: python score.py            -> score.json (frozen rule, both budgets, every model; determinism + CPU time)
       python score.py rules 32   -> rules.json (32 alternative fixed rules, clean models only; robustness study)"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

from frozen_check import E  # noqa: E402
import models as M  # noqa: E402
import truth as TR  # noqa: E402

BUDGETS = (16_000, 64_000)


def sampling_sec_per_fpe(model, budget, reps=None):
    """CPU s per FPE of plain vectorised sampling at this batch size (own forward, single thread)."""
    rng = np.random.default_rng(0); reps = reps or max(1, 400_000 // budget); t0 = time.process_time()
    for _ in range(reps):
        TR.n_correct(model, rng.standard_normal((budget, model[2].shape[0])))
    return (time.process_time() - t0) / (reps * budget)


def ratio(est, cost, tr):
    p, se = tr["acc"], tr["se"]
    return ((est - p) ** 2 - se ** 2) / (p * (1 - p) / cost)


def main():
    truths = json.load(open("truths.json")); ms = M.all_models(); out = {}
    spf = {b: sampling_sec_per_fpe(ms["f100"], b) for b in BUDGETS}
    for name, m in ms.items():
        for b in BUDGETS:
            t0 = time.process_time(); e, c = E.est(m, b, np.random.default_rng(1)); cpu = time.process_time() - t0
            e2, c2 = E.est(m, b, np.random.default_rng(2))
            r = dict(est=e, cost=c, cpu_s=cpu, det_ok=(e == e2 and c == c2), time_ratio=cpu / (c * spf[b]))
            if name in truths:
                r["ratio"] = ratio(e, c, truths[name]); r["truth"] = truths[name]["acc"]
            out[f"{name}|{b}"] = r
            print(name, b, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(dict(sampling_sec_per_fpe=spf, runs=out), open("score.json", "w"), indent=1)


def rules(R):
    truths = json.load(open("truths.json")); out = {}
    for s in M.SEEDS:
        m = M.load(s); tr = truths[f"f{s}"]
        for b in BUDGETS:
            rr = [E.est(m, b, rule_seed=7000 + 13 * k) for k in range(R)]
            est = np.array([e for e, _ in rr]); cost = np.array([c for _, c in rr])
            se2 = (est - tr["acc"]) ** 2 - tr["se"] ** 2
            samp = tr["acc"] * (1 - tr["acc"]) / cost
            boot = np.random.default_rng(0).integers(0, R, (2000, R))
            g = samp[boot].mean(1) / np.maximum(se2[boot].mean(1), 1e-30)
            out[f"f{s}|{b}"] = dict(est=est.tolist(), cost=cost.tolist(), gain=float(samp.mean() / se2.mean()),
                                    gain_ci=[float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
                                    bias_z=float((est.mean() - tr["acc"]) / np.sqrt(est.var(ddof=1) / R + tr["se"] ** 2)))
            print(s, b, {k: v for k, v in out[f"f{s}|{b}"].items() if k not in ("est", "cost")}, flush=True)
    json.dump(out, open("rules.json", "w"), indent=1)


if __name__ == "__main__":
    rules(int(sys.argv[2])) if len(sys.argv) > 1 and sys.argv[1] == "rules" else main()

"""T300 stage 0: the frozen T297 TB designs vs DEV2 1e8 truths on all 80 retrains. -> results/stage0.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402
import tb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
tr = r0.truths()["truths"]; ms = r0.dev2()
V = {"1752": dict(n23=1, newton=1, nblind=0, use12=False), "2472": dict(n23=1, newton=1, nblind=1, use12=False),
     "3192": dict(n23=1, newton=1, nblind=0, use12=True), "3912": dict(n23=1, newton=1, nblind=1, use12=True)}
out = {}
for key, v in V.items():
    rows = []
    for nm, model in ms:
        a, fl = tb.est(model, **v); p = tr[nm]["acc"]
        rows.append(dict(model=nm, est=a, truth=p, err=a - p, fpe=fl / r0.FPE, pq=p * (1 - p)))
    err = np.array([r["err"] for r in rows]); C = np.mean([r["fpe"] for r in rows]); pq = np.mean([r["pq"] for r in rows])
    rms = float(np.sqrt(np.mean(err ** 2) - np.mean([tr[r["model"]]["se"] ** 2 for r in rows])))
    F = rms ** 2 * C / pq
    # cluster bootstrap over models for F
    b = np.random.default_rng(0); Fs = []
    for _ in range(2000):
        ii = b.integers(len(rows), size=len(rows)); Fs.append(np.mean(err[ii] ** 2) * C / pq)
    lo, hi = np.percentile(Fs, [2.5, 97.5])
    out[key] = dict(variant=v, C_fpe=float(C), bias_pp=float(100 * err.mean()), rms_pp=100 * rms, F=float(F), F_ci=[float(lo), float(hi)],
                    lenient_threshold_rms_pp=float(100 * np.sqrt(10 * pq / C)), rows=rows)
    print(key, "C %.0f FPE  bias %+.2f pp  RMS %.2f pp  F %.2f [%.2f, %.2f]  (lenient RMS limit %.2f pp)" % (
        C, 100 * err.mean(), 100 * rms, F, lo, hi, out[key]["lenient_threshold_rms_pp"]), flush=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "stage0.json"), "w"), indent=1)

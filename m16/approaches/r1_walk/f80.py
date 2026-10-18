"""T304 stage B: lenient point F of the R1W variants on all 80 DEV2 retrains. -> results/f80.json"""
import json
import os
import sys

import numpy as np

import r1w
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

tr = r0.truths()["truths"]; out = {}
for name, v in (("V1", dict(K=1)), ("V2", dict(K=2)), ("V3", dict(K=2, blind12=True))):
    err, C, pq = [], [], []
    for nm, m in r0.dev2():
        a, f = r1w.est(m, **v); p = tr[nm]["acc"]; err.append(a - p); C.append(f / r0.FPE); pq.append(p * (1 - p))
    err = np.array(err); Cm = float(np.mean(C)); rms = float(np.sqrt(np.mean(err ** 2))); F = rms ** 2 * Cm / np.mean(pq)
    b = np.random.default_rng(0); Fs = [np.mean(err[b.integers(80, size=80)] ** 2) * Cm / np.mean(pq) for _ in range(2000)]
    out[name] = dict(variant=v, C_fpe=Cm, bias_pp=float(100 * err.mean()), rms_pp=100 * rms, F=float(F),
                     F_ci=[float(x) for x in np.percentile(Fs, [2.5, 97.5])], per_model_err_pp=(100 * err).tolist())
    print(name, v, "C %.0f  bias %+.3f  RMS %.3f  F %.2f %s" % (Cm, 100 * err.mean(), 100 * rms, F,
          [round(x, 2) for x in out[name]["F_ci"]]), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "f80.json"), "w"), indent=1)

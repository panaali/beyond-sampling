"""T302: lenient point F on all 80 DEV2 retrains for given walk lengths. -> results_f80.json"""
import json
import os
import sys

import numpy as np

import walk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

tr = r0.truths()["truths"]; out = {}
for tmax in [float(a) for a in sys.argv[1:]]:
    err, C, pq = [], [], []
    for nm, m in r0.dev2():
        a, f = walk.est(m, tmax=tmax); p = tr[nm]["acc"]; err.append(a - p); C.append(f / r0.FPE); pq.append(p * (1 - p))
    err = np.array(err); rms = float(np.sqrt(np.mean(err ** 2))); Cm = float(np.mean(C)); F = rms ** 2 * Cm / np.mean(pq)
    b = np.random.default_rng(0); Fs = [np.mean(err[b.integers(80, size=80)] ** 2) * Cm / np.mean(pq) for _ in range(2000)]
    out[str(tmax)] = dict(C_fpe=Cm, bias_pp=float(100 * err.mean()), rms_pp=100 * rms, F=float(F), F_ci=[float(x) for x in np.percentile(Fs, [2.5, 97.5])])
    print(tmax, {k: (round(v, 3) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in out[str(tmax)].items()}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_f80.json"), "w"), indent=1)

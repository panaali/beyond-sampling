"""T315: CAP-c on all 80 DEV2 (development), c = 1, 2, 3; per-region split vs 1e7 MC of R0 not applicable (closure c),
so the split is reported as estimator correct mass per region. -> results.json"""
import json
import os
import sys

import numpy as np

import cap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

tr = r0.truths()["truths"]; out = {}
for c in (1, 2, 3):
    err, C, pq, sp = [], [], [], []
    for nm, m in r0.dev2():
        a, f, s = cap.est(m, c=c, parts=True); p = tr[nm]["acc"]
        err.append(a - p); C.append(f / 5948); pq.append(p * (1 - p)); sp.append(s)
    err = np.array(err); Cm = float(np.mean(C)); rms = float(np.sqrt(np.mean(err ** 2))); F = rms ** 2 * Cm / np.mean(pq)
    b = np.random.default_rng(0); Fs = [np.mean(err[b.integers(80, size=80)] ** 2) * Cm / np.mean(pq) for _ in range(2000)]
    out[f"c{c}"] = dict(C_fpe_5948=Cm, bias_pp=float(100 * err.mean()), rms_pp=100 * rms, F=float(F),
                       F_ci=[float(x) for x in np.percentile(Fs, [2.5, 97.5])],
                       mean_correct_mass={k: float(np.mean([s[k] for s in sp])) for k in sp[0]})
    print("c", c, "C %.0f FPE  bias %+.3f  RMS %.3f  F %.2f %s" % (Cm, 100 * err.mean(), 100 * rms, F,
          [round(x, 2) for x in out[f"c{c}"]["F_ci"]]), flush=True)
out["winner"] = min((k for k in out), key=lambda k: out[k]["F"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
print("winner", out["winner"])

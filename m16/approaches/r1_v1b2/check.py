"""T313: (a) V1-B2 estimates == frozen V1 (r1_walk/estimator.est) to <= 1e-12 on all 80 DEV2; (b) instrumented C per
model (FLOPs, and FPE at 5,760 and 5,948 FLOPs/FPE); (c) frozen V1's charged FLOPs for comparison. -> results.json"""
import json
import os
import sys

import numpy as np

import v1b2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r1_walk"))
import estimator as V1  # noqa: E402

rows = {}
for nm, m in v1b2.r0.dev2():
    a1, f1 = V1.est(m); a2, f2 = v1b2.est(m)
    rows[nm] = dict(est_v1=a1, est_v1b2=a2, absdiff=abs(a1 - a2), flops_v1_frozen=f1, flops_v1b2=f2,
                    fpe_5760=f2 / 5760, fpe_5948=f2 / 5948)
    print(nm, "diff %.1e  V1 %.3e  V1-B2 %.3e FLOPs  (%.0f FPE @5948)" % (abs(a1 - a2), f1, f2, f2 / 5948), flush=True)
d = np.array([r["absdiff"] for r in rows.values()]); f = np.array([r["flops_v1b2"] for r in rows.values()])
s = dict(max_absdiff=float(d.max()), all_within_1e12=bool((d <= 1e-12).all()), mean_flops_v1b2=float(f.mean()),
         mean_fpe_5948=float(f.mean() / 5948), mean_fpe_5760=float(f.mean() / 5760),
         mean_flops_v1_frozen=float(np.mean([r["flops_v1_frozen"] for r in rows.values()])))
print(s)
json.dump(dict(summary=s, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)

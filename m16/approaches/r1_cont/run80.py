"""T317 steps 2-4 on all 80 DEV2: |est - V1| (G=48), |est(G=16) - est(G=48)|, instrumented C. -> results.json"""
import json
import os
import sys

import numpy as np

import cont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r1_walk"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r1w2  # noqa: E402
import r0  # noqa: E402

tr = r0.truths()["truths"]; rows = {}
for nm, m in r0.dev2():
    v1, _ = r1w2.est(m); a48, f48 = cont.est(m, G=48); a16, f16 = cont.est(m, G=16)
    rows[nm] = dict(v1=v1, cont48=a48, cont16=a16, d48=abs(a48 - v1), d16=abs(a16 - a48), C48=f48 / 5948, C16=f16 / 5948,
                    err48=a48 - tr[nm]["acc"])
d48 = max(r["d48"] for r in rows.values()); d16 = max(r["d16"] for r in rows.values())
C48 = float(np.mean([r["C48"] for r in rows.values()])); C16 = float(np.mean([r["C16"] for r in rows.values()]))
err = np.array([r["err48"] for r in rows.values()]); pq = np.mean([tr[n]["acc"] * (1 - tr[n]["acc"]) for n in rows])
s = dict(max_diff_vs_V1=d48, step2_pass=bool(d48 <= 1e-9), max_diff_G16_vs_G48=d16, G16_allowed=bool(d16 <= 1e-6),
         C48_fpe=C48, C16_fpe=C16, dev2_rms_pp=float(100 * np.sqrt(np.mean(err ** 2))),
         dev2_F_G48=float(np.mean(err ** 2) * C48 / pq), dev2_F_G16=float(np.mean(err ** 2) * C16 / pq))
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)

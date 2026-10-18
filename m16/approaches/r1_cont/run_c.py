"""T317c: equality vs T317b (<= 1e-12) and honest-cost saving on all 80 DEV2. -> results_c.json"""
import json
import os
import sys

import numpy as np

import cont_c
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

rows = {}
for nm, m in r0.dev2():
    ab, fb = cont_c.honest_baseline_cost(m); ac, fc = cont_c.est(m)
    rows[nm] = dict(diff=abs(ab - ac), C_b_honest=fb / 5948, C_c=fc / 5948)
d = max(r["diff"] for r in rows.values()); Cb = float(np.mean([r["C_b_honest"] for r in rows.values()]))
Cc = float(np.mean([r["C_c"] for r in rows.values()]))
s = dict(max_diff=d, equal_1e12=bool(d <= 1e-12), C_T317b_honest=Cb, C_T317c=Cc, saving_pct=100 * (1 - Cc / Cb),
         go=bool(d <= 1e-12 and (1 - Cc / Cb) >= 0.03))
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_c.json"), "w"), indent=1)

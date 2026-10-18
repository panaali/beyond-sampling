"""T322 gate on all 80 DEV2: (a) cont_d with T317c's MC points == cont_c (<= 1e-12; refactor check);
(b) cont_d (exact points) within .01 pp of T317c; cost as executed. -> results_d.json"""
import json
import os
import sys
import time

import numpy as np

import cont_c
import cont_d
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

rows = {}; t0 = time.time()
for nm, m in r0.dev2():
    ac, fc = cont_c.est(m); am, fm = cont_d.est(m, points="mc"); ad, fd = cont_d.est(m)
    rows[nm] = dict(est_c=ac, est_d=ad, diff_mc=abs(am - ac), dfl_mc=fm - fc, diff_pp=100 * (ad - ac),
                    C_c=fc / 5948, C_d=fd / 5948)
    print(nm, rows[nm], round(time.time() - t0), flush=True)
R = list(rows.values())
s = dict(max_diff_mc=max(r["diff_mc"] for r in R), max_dflops_mc=max(abs(r["dfl_mc"]) for r in R),
         max_abs_diff_pp=max(abs(r["diff_pp"]) for r in R), mean_diff_pp=float(np.mean([r["diff_pp"] for r in R])),
         C_c=float(np.mean([r["C_c"] for r in R])), C_d=float(np.mean([r["C_d"] for r in R])),
         tables=[list(map(str, t)) for t in cont_d.table_sizes()])
s["gate"] = bool(s["max_diff_mc"] <= 1e-12 and s["max_abs_diff_pp"] <= 0.01)
print(json.dumps({k: v for k, v in s.items() if k != "tables"}, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_d.json"), "w"), indent=1)

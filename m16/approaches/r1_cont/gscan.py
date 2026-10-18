"""T317b: G in {20, 24, 32} vs G = 48 on all 80 DEV2 (max |diff|, mean C). -> results_g.json"""
import json
import os

import numpy as np

import cont
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")))["rows"]
out = {}
for G in (20, 24, 32):
    d, C = [], []
    for nm, m in r0.dev2():
        a, f = cont.est(m, G=G); d.append(abs(a - base[nm]["cont48"])); C.append(f / 5948)
    out[G] = dict(max_absdiff=float(max(d)), max_diff_pp=float(100 * max(d)), ok=bool(max(d) <= 1e-6), C_fpe=float(np.mean(C)))
    print(G, out[G], flush=True)
q = [G for G in (20, 24, 32) if out[G]["ok"]]
out["chosen"] = min(q) if q else None
print("chosen", out["chosen"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_g.json"), "w"), indent=1)

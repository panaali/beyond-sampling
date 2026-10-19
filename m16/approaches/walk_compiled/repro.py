"""Reproduction gate: frozen est (numpy walk) vs est with the compiled walk (repro and fast CDF modes).
usage: python repro.py T [budgets]   -> repro_T{T}.json"""
import json
import sys
import time

import numpy as np

import models as Mo
import walk_nb as WN

E = Mo.frozen()
frozen_walk = E.W.walk
T = int(sys.argv[1])
budgets = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "16000,64000").split(",")]
out = {}; worst = 0.0
for name, m in Mo.groups(T).items():
    for B in budgets:
        E.W.walk = frozen_walk; t0 = time.perf_counter(); e0, c0 = E.est(m, B); tf = time.perf_counter() - t0
        row = {"frozen": [e0, c0], "t_frozen": tf}
        for mode in ("repro", "fast"):
            WN.MODE["fast"] = mode == "fast"; E.W.walk = WN.walk
            e1, c1 = E.est(m, B)
            re, rc = abs(e1 - e0) / abs(e0), abs(c1 - c0) / abs(c0)
            row[mode] = [e1, c1, re, rc]; worst = max(worst, re, rc)
        out[f"{name}:{B}"] = row
        print(name, B, f"est {e0:.12f} cost {c0:.1f} | repro rel {row['repro'][2]:.1e}/{row['repro'][3]:.1e}"
              f" | fast rel {row['fast'][2]:.1e}/{row['fast'][3]:.1e} | frozen {tf:.2f}s", flush=True)
E.W.walk = frozen_walk
print("worst relative difference", worst, "PASS" if worst <= 1e-12 else "FAIL")
json.dump({"worst": worst, "rows": out}, open(f"{Mo.HERE}/repro_T{T}.json", "w"), indent=1)

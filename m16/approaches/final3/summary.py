"""Collect exp1 table + T trend (H=8) -> summary.json. Gap ratio = (recipe - estT exact)^2 / sampling MSE, i.e. the
certificate's own error if the shared leaf-overlap mass defect were fixed."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43"); sys.path[:0] = [U, os.path.join(HERE, "..", "..")]
from zoo44 import zoo
from run_regions import regions
from leaderboard import fpe_flops
R = json.load(open(os.path.join(HERE, "res_H8_T4.json"))); E = json.load(open(os.path.join(HERE, "exact_H8_T4.json")))
e3 = json.load(open(os.path.join(U, "evalH_8.json")))["zoo"]; C = json.load(open(os.path.join(HERE, "cells_nomass.json")))
S = {"T4": {}, "T3": {}, "T5": {}}
for k, r in R.items():
    smse = r["truth"] * (1 - r["truth"]) / r["N_equiv"]; d = dict(r)
    if k in E:
        d.update(exact=E[k], gap_ratio=(r["est"] - E[k]["acc"]) ** 2 / smse)
    S["T4"][k] = d
for s in range(5):
    S["T3"][f"zoo{s}"] = dict(cells=e3[str(s)]["clean_cells"], ops=e3[str(s)]["clean_ops"], regions_exact=e3[str(s)]["regions"],
                             regions_lb=regions(zoo(8, 3, s), 3))
for k, v in C.items():
    if k.startswith("H8_T5"):
        S["T5"][k] = v
json.dump(S, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
for T in ("T3", "T4", "T5"):
    z = [v for k, v in S[T].items() if "noisy" not in k]
    if z:
        print(T, "cells", np.mean([v["cells"] for v in z]).round(), "regions_lb", np.mean([v["regions_lb"] for v in z]).round(),
              "ops", "%.3g" % np.mean([v.get("ops", v.get("ops_nomass", 0)) for v in z]))
print("T4 gap ratios", [round(v["gap_ratio"], 2) for v in S["T4"].values() if "gap_ratio" in v])

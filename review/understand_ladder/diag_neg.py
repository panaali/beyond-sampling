"""T97: diagnose the NEGATIVE F2L2 error (estimate too low = error mass over-counted) on the adversarial H=6 model:
per-family F2L2 vs FX (author's exact local trace) vs reviewer's exact truth. usage: python diag_neg.py H"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder"))
import fibre  # noqa: E402
import rtruth as R  # noqa: E402

H = int(sys.argv[1])
J = json.load(open(os.path.join(HERE, f"adv_H{H}.json")))
m = tuple(np.array(w) for w in J["worst"]["model"])
ex = R.exact(m)[0]
out = {}
for L in (1, 2, 3):
    fibre.LOOK = L; inf = {}; e = fibre.est_fibre2(m, fibre.Ops(), inf); out[f"F2L{L}"] = (e, inf["err_by"])
inf = {}; e = fibre.est_fibre(m, "FX", fibre.Ops(), inf); out["FX"] = (e, inf["err_by"])
for k, (e, _) in out.items():
    print(f"{k:5s} est {e:.6f} err {e - ex:+.2e}")
print("per family (error mass): FX, F2L1, F2L2, F2L3")
for key in out["FX"][1]:
    v = [out[k][1][key] for k in ("FX", "F2L1", "F2L2", "F2L3")]
    if max(v) - min(v) > 1e-9:
        print(key, " ".join(f"{x:.6f}" for x in v))

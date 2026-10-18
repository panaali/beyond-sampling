"""T370 validation: closure='R1W' through t370's per-node loop must reproduce T366's E1 on s547 (results_t369_R1) and the
fixed-walk K3 E2 (T354 S3, T_hi 6) on orig. -> results_t370_val.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t370 as Z  # noqa: E402

z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
i = names.index("dev3_s547"); m547 = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
r1 = json.load(open(os.path.join(HERE, "results_t369_R1.json")))["dev3_s547"]["est"]
e, f = Z.est_e1(m547, (2.25, 2.5), None, closure="R1W")
out = {"E1_s547": {"t370_closure": e, "t366": r1, "rel": {str(k): e[k] / r1[str(k)] - 1 for k in e}}}
print(out, flush=True)
fx = json.load(open(os.path.join(HERE, "results_fixed_B_C3_S3.json")))["orig:S3"]["est2"]
orig = Z.TC.cf.lb3_models()["orig"]
for t in (0.5, 0.75):
    e2, _ = Z.est_e2(orig, t, None, closure="R1W", tmax=6.0)
    out[f"E2_orig_{t}"] = dict(t370_closure=e2, s3=fx[str(t)], rel=e2 / fx[str(t)] - 1); print(t, out[f"E2_orig_{t}"], flush=True)
json.dump(out, open(os.path.join(HERE, "results_t370_val.json"), "w"), indent=1, default=float)

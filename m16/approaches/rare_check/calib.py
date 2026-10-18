"""T349 calibration: my estimators on T340's 3 models vs T192's GPU truths (t340/rare_mc.json @63d36d38)."""
import json
import numpy as np
import rare

z = np.load("/home/user/dl/t340/weights.npz", allow_pickle=True)
TR = json.load(open("/home/user/dl/t340/rare_mc.json"))
key1 = {1.0: "E1_1", 1.25: "E1_1p25", 1.5: "E1_1p5", 0.0: "E1_0"}; key2 = {0.5: "E2_0p5", 0.75: "E2_0p75"}
out = {}
for i, nm in enumerate(z["names"]):
    m = tuple(np.asarray(z[k][i], float) for k in ("Wi", "Wh", "Wo")); nm = str(nm)
    e1, c1 = rare.est_E1(m); e2, c2 = rare.est_E2(m); n = TR[nm]["n"]
    rows = {}
    for t0, k in key1.items():
        p = TR[nm]["counts"][k] / n; rows[k] = dict(est=e1[t0], truth=p, ratio=e1[t0] / p, C=c1)
    for t, k in key2.items():
        p = TR[nm]["counts"][k] / n; rows[k] = dict(est=e2[t], truth=p, ratio=e2[t] / p, C=c2)
    out[nm] = rows
    print(nm, {k: round(v["ratio"], 3) for k, v in rows.items()}, "C_E1", round(c1), "C_E2", round(c2), flush=True)
json.dump(out, open("results/calib.json", "w"), indent=1)

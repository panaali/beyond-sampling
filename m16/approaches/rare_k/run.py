"""T360: K=3 and K=5 estimates on T340's 3 models (calibration) and dev3_s500..525. Estimates only."""
import json
import numpy as np
import rare_k as RK

out = {}
jobs = [("/home/user/dl/t340/weights.npz", ["orig", "seed0", "seed3"], True)]
jobs.append(("/home/user/dl/dev3/weights.npz", [f"dev3_s{i}" for i in range(500, 506)], True))
jobs.append(("/home/user/dl/dev3/weights.npz", [f"dev3_s{i}" for i in range(506, 526)], False))
for path, names, do_e1 in jobs:
    z = np.load(path, allow_pickle=True); allnames = [str(n) for n in z["names"]]
    for nm in names:
        i = allnames.index(nm); m = tuple(np.asarray(z[k][i], float) for k in ("Wi", "Wh", "Wo"))
        out[nm] = {}
        for K in (3, 5):
            e1, c1, e2, c2 = RK.est(m, K, t0s=(1.0, 1.25, 1.5) if do_e1 else ())
            out[nm][f"K{K}"] = {"E1": {str(k): float(v) for k, v in e1.items()}, "C_E1": float(c1),
                                "E2": {str(k): float(v) for k, v in e2.items()}, "C_E2": float(c2)}
        print(nm, json.dumps(out[nm]), flush=True)
        json.dump(out, open("results/est.json", "w"), indent=1)

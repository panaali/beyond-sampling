"""M5t on T398 survivor files (certs/surv_d<delta>_<tri>_k<k>.npz: tri, k, gates (N,10,16), delta): rigorous
[lb, ub] on gamma(cell ∩ N_delta ∩ {g23 >= t0}) per cell (event ignored), with N_delta = {g23 >= delta ||x||_inf}
(delta's float64 value as an exact rational). Shard by [i0, i1).
usage: python surv_mass.py FILE t0 budget i0 i1 -> results_surv_<name>_<t0>_<i0>.json"""
import json
import os
import sys
import time

import numpy as np

import rigour as R

f, t0, bud, i0, i1 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
c = np.load(f, allow_pickle=True)
tri = tuple(int(v) for v in c["tri"]); k = int(c["k"]); norm = 1 / R.q(float(c["delta"]))
out = []
for i in range(i0, min(i1, len(c["gates"]))):
    t = time.time()
    try:
        lb, ub, info = R.mass_cell(tri, c["gates"][i].astype(int), k, t0, budget=bud, return_info=True, tangent=True,
                                   zbins=24, zratio=R.fmpq(23, 20), norm=norm)
        err = None
    except Exception as e:                                  # reported, never silently dropped
        lb, ub, info, err = 0.0, float("inf"), {}, repr(e)
    out.append(dict(i=i, lb=lb, ub=ub, t=time.time() - t, simplices=info.get("simplices"), err=err))
    print(out[-1], flush=True)
name = os.path.basename(f)[:-4]
json.dump(out, open(f"results_surv_{name}_{t0}_{i0}.json", "w"), indent=0)
print("SUM ub %.6g lb %.6g cells %d errors %d t_median %.2f" % (sum(o["ub"] for o in out), sum(o["lb"] for o in out),
      len(out), sum(o["err"] is not None for o in out), np.median([o["t"] for o in out])))

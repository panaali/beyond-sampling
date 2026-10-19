"""T416 cost/tightness sample: mass_cell_fast (T412) on k = 10 survivors with the event rows.
usage: python k10_sample.py SURV_FILE t0 budget i0 i1 [step] -> results_k10sample_<name>_<t0>_<budget>_<i0>.json"""
import json
import os
import sys
import time

import numpy as np

import m5t_fast as F
import rigour as R

f, t0, bud, i0, i1 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
step = int(sys.argv[6]) if len(sys.argv) > 6 else 1
c = np.load(f, allow_pickle=True); tri = tuple(int(v) for v in c["tri"]); norm = 1 / R.q(float(c["delta"]))
out = []
for i in range(i0, min(i1, len(c["gates"])), step):
    t = time.time()
    try:
        lb, ub, info = F.mass_cell_fast(tri, c["gates"][i].astype(int), 10, t0, budget=bud, norm=norm, event=True, return_info=True)
        err = None
    except Exception as e:
        lb, ub, info, err = 0.0, float("inf"), {}, repr(e)
    rec = dict(i=i, lb=lb, ub=ub, t=time.time() - t, err=err, **{k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str))})
    out.append(rec); print(rec, flush=True)
json.dump(out, open(f"results_k10sample_{os.path.basename(f)[:-4]}_{t0}_{bud}_{i0}.json", "w"), indent=0)
print("SUM ub %.4g lb %.4g n %d t_med %.1f" % (sum(o["ub"] for o in out), sum(o["lb"] for o in out), len(out), np.median([o["t"] for o in out])))

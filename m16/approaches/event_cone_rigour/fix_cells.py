"""Re-run tier 2 (fast M5t, fast geometry) on given cells of a tiered10 result file whose M5t errored, and update
that file in place: ub = min(M2, M5t), lb = M5t lb, err_m5t replaced by fixed_from.
usage: python fix_cells.py RESULT_JSON SURV_FILE t0 i [i ...]"""
import json
import sys
import time

import numpy as np

import geom_fast as GF
import m5t_fast as F
import rigour as R

rj, f, t0 = sys.argv[1], sys.argv[2], float(sys.argv[3])
idx = [int(v) for v in sys.argv[4:]]
c = np.load(f, allow_pickle=True); tri = tuple(int(v) for v in c["tri"]); norm = 1 / R.q(float(c["delta"]))
out = json.load(open(rj)); by = {o["i"]: o for o in out}
for i in idx:
    rec = by[i]; g = c["gates"][i].astype(int); t = time.time()
    gm = GF.cell_geometry_fast(tri, g, 10, norm=norm, event=True)
    lb, ub, info = F.mass_cell_fast(tri, g, 10, t0, budget=4000, norm=norm, event=True, return_info=True, geom=gm)
    rec["fixed_from"] = rec.pop("err_m5t", None)
    rec.update(ub=min(rec["m2"], ub), lb=lb, tier=2, vertices=info.get("vertices"), simplices=info.get("simplices"),
               geom="fast", t_m5t=time.time() - t)
    print(rec, flush=True)
json.dump(out, open(rj, "w"), indent=0)

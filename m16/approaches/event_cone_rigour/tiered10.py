"""T416: rigorous charging of k = 10 survivors (event rows exact). Per cell: tier (i) M2 (rigour.mass_upper,
event rows, N_delta); if its ub < CAP the cell is charged with it; otherwise tier (ii) m5t_fast.mass_cell_fast
(budget B), charged with min(M2, M5t). Every charge is a rigorous upper bound on gamma(cell ∩ E(t0) ∩ N_delta).
usage: python tiered10.py SURV_FILE t0 i0 i1 [CAP=1e-16] [B=4000] -> results_t10_<name>_<t0>_<i0>.json
(PROVISIONAL until T417, the independent audit of m5t_fast, reports.)"""
import json
import os
import sys
import time

import numpy as np

import m5t_fast as F
try:
    import geom_fast as GF                      # T415 (identical outputs, faster); fallback: m5t_fast's geometry
except Exception:
    GF = None
import rigour as R

f, t0, i0, i1 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
CAP = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-16
B = int(sys.argv[6]) if len(sys.argv) > 6 else 4000
c = np.load(f, allow_pickle=True); tri = tuple(int(v) for v in c["tri"]); norm = 1 / R.q(float(c["delta"]))
name = os.path.basename(f)[:-4]
path = f"results_t10_{name}_{t0}_{i0}.json"
out = json.load(open(path)) if os.path.exists(path) else []          # resumable
done = {o["i"] for o in out}
for i in range(i0, min(i1, len(c["gates"]))):
    if i in done:
        continue
    g = c["gates"][i].astype(int); rec = dict(i=i)
    t = time.time()
    try:
        rec["m2"] = R.mass_upper(tri, g, 10, t0, norm=norm)
    except Exception as e:
        rec["m2"] = float("inf"); rec["err_m2"] = repr(e)
    rec["t_m2"] = time.time() - t
    rec["ub"] = rec["m2"]; rec["lb"] = 0.0; rec["tier"] = 1
    if rec["m2"] >= CAP:
        t = time.time()
        try:
            gm = GF.cell_geometry_fast(tri, g, 10, norm=norm, event=True) if GF is not None else None
            lb, ub, info = F.mass_cell_fast(tri, g, 10, t0, budget=B, norm=norm, event=True, return_info=True,
                                            **({} if gm is None else dict(geom=gm)))
            rec.update(ub=min(rec["m2"], ub), lb=lb, tier=2, vertices=info.get("vertices"), simplices=info.get("simplices"),
                       geom="fast" if gm is not None else "ref")
        except Exception as e:
            rec["err_m5t"] = repr(e)
        rec["t_m5t"] = time.time() - t
    out.append(rec); print(rec, flush=True)
    if len(out) % 25 == 0:
        json.dump(out, open(path, "w"), indent=0)
json.dump(out, open(path, "w"), indent=0)
print("DONE cells %d sum_ub %.4g sum_lb %.4g tier2 %d errors %d" % (len(out), sum(o["ub"] for o in out), sum(o["lb"] for o in out),
      sum(o["tier"] == 2 for o in out), sum(("err_m2" in o and o["tier"] == 1) or "err_m5t" in o for o in out)))

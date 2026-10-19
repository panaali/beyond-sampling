"""T452 profiling: per tier-2 cell of T400's (9,3,2) sizing sample, time geom_fast.cell_geometry_fast by stage and
m5t_fast.mass_cell_fast (the tiered10.py tier-2 call). usage: python prof.py -> results_prof.json"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ECR = os.path.join(HERE, "..", "event_cone_rigour")
sys.path.insert(0, ECR)
import geom_fast as GF  # noqa: E402
import m5t_fast as F  # noqa: E402
import rigour as R  # noqa: E402

SURV = os.path.join(HERE, "data", "surv_dfs_d0.333_9-3-2_k4to10_s0of6.npz")


def tier2_sample():
    idx = []
    for f in sorted(glob.glob(os.path.join(ECR, "results_t10sample_surv_dfs_d0.333_9-3-2_k4to10_s0of6_2.5_*_step71.json"))):
        idx += [o["i"] for o in json.load(open(f)) if o.get("tier") == 2]
    return sorted(idx)


STAGES = [(R, "exact_rows"), (R, "irredundant"), (R.Poly, "start"), (GF, "enumerate_fast"), (GF, "_float_vertices"),
          (GF, "pulling_triangulation_fast"), (GF, "volumes_iv"), (GF, "q2iv_ref")]


def instrument(tm):
    for obj, name in STAGES:
        o = getattr(obj, name)

        def w(*a, _o=o, _n=name, **k):
            t = time.perf_counter(); r = _o(*a, **k); tm[_n] = tm.get(_n, 0.0) + time.perf_counter() - t; return r
        setattr(obj, name, w)


if __name__ == "__main__":
    c = np.load(SURV, allow_pickle=True); tri = tuple(int(v) for v in c["tri"]); norm = 1 / R.q(float(c["delta"]))
    tm = {}; instrument(tm)
    rows = []
    for i in tier2_sample():
        g = c["gates"][i].astype(int)
        tm.clear()
        t = time.perf_counter()
        try:
            gm = GF.cell_geometry_fast(tri, g, 10, norm=norm, event=True)
        except MemoryError:
            rows.append(dict(i=i, error="MemoryError", t_geom=time.perf_counter() - t, t_mass=0.0, stages=dict(tm)))
            print(rows[-1], flush=True); continue
        tg = time.perf_counter() - t
        st = dict(tm)
        t = time.perf_counter()
        r = F.mass_cell_fast(tri, g, 10, 2.5, budget=4000, norm=norm, event=True, return_info=True, geom=gm) if gm else None
        tb = time.perf_counter() - t
        rows.append(dict(i=i, t_geom=tg, t_mass=tb, stages=st, vertices=gm[-1] if gm else 0, simplices0=len(gm[5]) if gm else 0,
                         ub=r[1] if r else None))
        print(rows[-1], flush=True)
    s = dict(cells=len(rows), t_geom=sum(r["t_geom"] for r in rows), t_mass=sum(r["t_mass"] for r in rows),
             stages={k: sum(r["stages"].get(k, 0) for r in rows) for k in rows[0]["stages"]})
    print(json.dumps(s, indent=1))
    json.dump(dict(summary=s, cells=rows), open(os.path.join(HERE, "results_prof.json"), "w"), indent=0)

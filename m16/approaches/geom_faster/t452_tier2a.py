"""T452 addendum measurement (README 'Addendum'): tier 2a on T400's 201-cell (9,3,2) sizing sample.
usage: python t452_tier2a.py            (driver: a fresh process per tier-2 cell) -> results_tier2a.json
       python t452_tier2a.py one I      (one cell; appends to data/tier2a_cells.jsonl)"""
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ECR = os.path.join(HERE, "..", "event_cone_rigour")
sys.path.insert(0, ECR); sys.path.insert(0, HERE)
import geom_faster as G2  # noqa: E402
import mass_screened as MS  # noqa: E402
import rigour as R  # noqa: E402

sample = []
for f in sorted(glob.glob(os.path.join(ECR, "results_t10sample_surv_dfs_d0.333_9-3-2_k4to10_s0of6_2.5_*_step71.json"))):
    sample += json.load(open(f))
ref = {r["i"]: r for r in json.load(open(os.path.join(HERE, "results_speed.json")))["cells"]}
c = np.load(os.path.join(HERE, "data", "surv_dfs_d0.333_9-3-2_k4to10_s0of6.npz"), allow_pickle=True)
tri = tuple(int(v) for v in c["tri"]); norm = 1 / R.q(float(c["delta"]))
CELLS = os.path.join(HERE, "data", "tier2a_cells.jsonl")


def one(i):
    """One tier-2 cell in this (fresh) process; appends its record to CELLS."""
    o = next(x for x in sample if x["i"] == i); r = ref.get(i, {})
    G2.cell_geometry_faster(tri, c["gates"][284].astype(int), 10, norm=norm, event=True)    # compile (not timed)
    g = c["gates"][i].astype(int)
    t = time.perf_counter()
    try:
        gm = G2.cell_geometry_faster(tri, g, 10, norm=norm, event=True)
        lb, ub, used, B = MS.tier2a(tri, g, 10, 2.5, o["m2"], norm=norm, event=True, geom=gm)
        err = None
    except MemoryError:
        lb, ub, used, B, err = 0.0, o["m2"], False, None, "MemoryError"     # tiered10 charges M2 on failure
    tn = time.perf_counter() - t
    ub_id = min(o["m2"], r["ub"]) if r.get("ub") is not None else o["ub"]
    rec = dict(i=i, tier=2, used_2a=used, B=B, charge_ref=ub_id, charge_new=ub, t_new=tn, err=err,
               t_ref=(r.get("t_geom_ref", 0) + r.get("t_mass_ref", 0)) if r.get("err_ref") is None else None)
    with open(CELLS, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    print(rec, flush=True)


if len(sys.argv) > 2 and sys.argv[1] == "one":
    one(int(sys.argv[2])); sys.exit()
import subprocess  # noqa: E402
done = {json.loads(l)["i"] for l in open(CELLS)} if os.path.exists(CELLS) else set()
for o in sample:                                        # a fresh process per cell: no address-space growth
    if o["tier"] == 2 and o["i"] not in done:
        subprocess.run([sys.executable, os.path.abspath(__file__), "one", str(o["i"])])
got = {}
for l in open(CELLS):
    x = json.loads(l); got[x["i"]] = x
rows = [got[o["i"]] if o["tier"] == 2 else dict(i=o["i"], tier=1, charge_ref=o["ub"], charge_new=o["ub"]) for o in sample]
t2 =[x for x in rows if x["tier"] == 2 and x.get("t_ref") is not None and x["err"] is None]
s = dict(cells=len(rows), tier2=sum(x["tier"] == 2 for x in rows), tier2a=sum(x.get("used_2a", False) for x in rows),
         ub_new_ge_ref=sum(x["charge_new"] >= x["charge_ref"] for x in rows if x["tier"] == 2),
         ub_new_lt_ref=[(x["i"], x["charge_new"], x["charge_ref"]) for x in rows if x["tier"] == 2 and x["charge_new"] < x["charge_ref"]][:10],
         total_ref=math.fsum(x["charge_ref"] for x in rows), total_new=math.fsum(x["charge_new"] for x in rows),
         t2_time_ref=sum(x["t_ref"] for x in t2), t2_time_new=sum(x["t_new"] for x in t2), t2_compared=len(t2))
s["speedup_tier2"] = s["t2_time_ref"] / s["t2_time_new"]
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, cells=rows), open(os.path.join(HERE, "results_tier2a.json"), "w"), indent=0)

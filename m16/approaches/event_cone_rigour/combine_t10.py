"""Merge rule for T416 (tiered10 shards): every survivor index 0..N-1 of each file appears exactly once; a cell
with an unresolved error is charged inf; report Σub, Σlb, tiers, errors, top cells, geometry path.
usage: python combine_t10.py SURV_PREFIX t0   (e.g. surv_d0.333_1-8-2_k10 2.5)"""
import glob
import json
import sys

import numpy as np

pre, t0 = sys.argv[1], sys.argv[2]
tot_ub = tot_lb = 0.0; allrecs = []
for s in (0, 1):
    N = len(np.load(f"data/certs/{pre}_s{s}.npz", allow_pickle=True)["gates"])
    seen = {}
    for f in sorted(glob.glob(f"results_t10_{pre}_s{s}_{t0}_*.json")):
        for o in json.load(open(f)):
            if o["i"] in seen:
                print("DUPLICATE", s, o["i"], f)
            seen[o["i"]] = o
    missing = sorted(set(range(N)) - set(seen))
    print(f"s{s}: N {N} covered {len(seen)} missing {len(missing)}", missing[:10])
    for i, o in seen.items():
        ub = o["ub"] if np.isfinite(o["ub"]) else float("inf")
        if "err_m2" in o and o.get("tier") == 1:
            ub = float("inf")
        allrecs.append((ub, o.get("lb", 0.0), s, i, o.get("tier"), o.get("geom", "ref/unknown"), "err_m5t" in o))
    if missing:
        tot_ub = float("inf")
ub = np.array([r[0] for r in allrecs]); lb = np.array([r[1] for r in allrecs])
print("cells %d  Σub %.4g  Σlb %.4g  tier2 %d  unresolved err_m5t (charged M2) %d  inf %d" % (
    len(allrecs), ub.sum() if tot_ub == 0 else tot_ub, lb.sum(), sum(r[4] == 2 for r in allrecs), sum(r[6] for r in allrecs), int(np.isinf(ub).sum())))
for r in sorted(allrecs, reverse=True)[:10]:
    print("  top: ub %.3g lb %.3g s%d i%d tier %s geom %s err %s" % r)

"""T373 projection and verdict from results_k6_fast_*.json (README, pre-registration)."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "results_k6_fast_*.json"))):
    R.update(json.load(open(f)))
rows = []
for tri, d in R.items():
    c = {r["k"]: r for r in d["counts"]}
    last = max(k for k in c if c[k]["complete"])
    n5 = c[5]["regions"] if 5 in c else None; n6 = c[6]["regions"] if 6 in c else None
    capped = not all(c[k]["complete"] for k in c) or last < 6
    if last >= 6:
        g = n6 / n5; term = n6 * g ** 4
    else:                                                          # censored: lower bound from the last complete step
        kk = last; g = c[kk]["regions"] / c[kk - 1]["regions"]
        n6 = c[kk]["regions"] * g ** (6 - kk) if not (kk + 1 in c) else max(c[kk + 1]["regions"], c[kk]["regions"])
        term = None
    ev = d.get("events", {})
    rows.append(dict(tri=tri, counts=[c[k]["regions"] for k in sorted(c)], complete=[c[k]["complete"] for k in sorted(c)],
                     last_complete=last, g56=g, term=term, cpu=d["cpu_sec"], ev=ev))
for r in rows:
    print(r)
print(json.dumps(rows, indent=1), file=open(os.path.join(HERE, "summary.json"), "w"))

"""A2/A3 for M5 (mass_cell) on the k=4 benchmark cells with >= 30 plain-MC hits. usage: python a_m5.py t0 budget"""
import json
import sys
import time

import numpy as np

import rigour as R

t0 = float(sys.argv[1]); bud = int(sys.argv[2])
d = np.load("data/bench_182_k4.npz", allow_pickle=True); tri = tuple(int(v) for v in d["tri"]); k = int(d["k"])
a = {o["i"]: o for o in json.load(open(f"results_a_cells_{t0}.json"))}
out = []
for i in sorted(i for i in a if a[i]["hits"] >= 30):
    g = np.full((10, 16), -1); g[:k] = np.unpackbits(np.frombuffer(d["hist"][i], np.uint8)).reshape(k, 16)
    t = time.time(); lb, ub, info = R.mass_cell(tri, g, k, t0, budget=bud, return_info=True)
    out.append(dict(i=i, lb=lb, ub=ub, mc=a[i]["mc"], hits=a[i]["hits"], m2=a[i]["ub"], t=time.time() - t,
                    vertices=info["vertices"], simplices=info["simplices"])); print(out[-1], flush=True)
json.dump(out, open(f"results_a_m5_{t0}_{bud}.json", "w"), indent=0)

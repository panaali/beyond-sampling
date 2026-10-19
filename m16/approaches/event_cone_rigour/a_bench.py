"""A2/A3 for the cheap call mass_upper on T398's k=4 benchmark cells (event ignored): rigorous ub vs MC, cost.
usage: python a_bench.py t0 n_mc i0 i1 -> results_a_bench_t0.json"""
import json
import sys
import time

import numpy as np

import mc
import rigour as R

d = np.load("data/bench_182_k4.npz", allow_pickle=True)
tri = tuple(int(v) for v in d["tri"]); k = int(d["k"]); t0 = float(sys.argv[1]); n = int(float(sys.argv[2]))
out = []
for i in range(int(sys.argv[3]), int(sys.argv[4])):
    g = np.full((10, 16), -1); g[:k] = np.unpackbits(np.frombuffer(d["hist"][i], np.uint8)).reshape(k, 16)
    t = time.time(); ub, info = R.mass_upper(tri, g, k, t0, return_info=True); tu = time.time() - t
    m, se, h = mc.mc_mass(tri, g, t0, n=n, s=1.0, k=k, event=False, seed=i)
    out.append(dict(i=i, ub=ub, mc=m, se=se, hits=h, t=tu, **info)); print(out[-1], flush=True)
json.dump(out, open(f"results_a_bench_{t0}_{sys.argv[3]}.json", "w"), indent=0)

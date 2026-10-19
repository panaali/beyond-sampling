"""A2/A3 for the cheap call mass_upper on T398's 273 k=4 benchmark cells (event ignored), against plain MC
(mc_cells.py counts). usage: python a_cells.py t0 -> results_a_cells_<t0>.json"""
import json
import sys
import time

import numpy as np

import rigour as R

t0 = float(sys.argv[1])
d = np.load("data/bench_182_k4.npz", allow_pickle=True)
tri = tuple(int(v) for v in d["tri"]); k = int(d["k"])
mc = np.load(f"data/mc_cells_{'-'.join(map(str, tri))}_k{k}_{t0}.npz", allow_pickle=True)
cnt = dict(zip(mc["keys"], mc["counts"])); n = float(mc["draws"])
out = []
for i in range(len(d["hist"])):
    h = bytes(d["hist"][i])
    g = np.full((10, 16), -1); g[:k] = np.unpackbits(np.frombuffer(h, np.uint8)).reshape(k, 16)
    t = time.time(); ub, info = R.mass_upper(tri, g, k, t0, return_info=True); tu = time.time() - t
    c = int(cnt.get(h, 0))
    out.append(dict(i=i, ub=ub, mc=c / (720 * n), hits=c, t=tu, S=info.get("S")))
    print(out[-1], flush=True)
json.dump(out, open(f"results_a_cells_{t0}.json", "w"), indent=0)
r = np.array([o["ub"] / o["mc"] for o in out if o["hits"] >= 30])
print("cells with >=30 hits", len(r), "ub/mc median %.3g p90 %.3g min %.3g" % (np.median(r), np.percentile(r, 90), r.min()),
      "violations(ub<mc)", int((r < 1).sum()), "t median %.3f" % np.median([o["t"] for o in out]),
      "sum ub %.3g sum mc %.3g" % (sum(o["ub"] for o in out), sum(o["mc"] for o in out)))

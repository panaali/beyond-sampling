"""T412: record every simplex that the arb M5t (rigour.mass_cell, tangent=True, a_m5.py settings) evaluates on the
k = 4 bench cells with >= 30 MC hits, with the arb [lo, hi] and the arb time per simplex. rigour.py is not edited:
_bounds_tangent is wrapped. usage: python t412_record.py t0 -> data/t412_rec_<t0>.npz"""
import json
import sys
import time

import numpy as np

import rigour as R
from m5t_fast import q2iv

t0 = float(sys.argv[1]); bud = 2000
d = np.load("data/bench_182_k4.npz", allow_pickle=True); tri = tuple(int(v) for v in d["tri"]); k = int(d["k"])
cells = [o for o in json.load(open(f"results_a_m5t_{t0}_2000.json"))]
rec = dict(cell=[], Mlo=[], Mhi=[], vlo=[], vhi=[], alo=[], ahi=[], t=[])
orig = R._bounds_tangent
cur = [None]


def wrapped(Ys, Ms, vol, *a):
    t = time.perf_counter(); lo, hi = orig(Ys, Ms, vol, *a); t = time.perf_counter() - t
    iv = [[q2iv(x) for x in m] for m in Ms]; vv = q2iv(vol)
    rec["cell"].append(cur[0]); rec["t"].append(t)
    rec["Mlo"].append([[x[0] for x in m] for m in iv]); rec["Mhi"].append([[x[1] for x in m] for m in iv])
    rec["vlo"].append(vv[0]); rec["vhi"].append(vv[1])
    rec["alo"].append(float(lo.lower())); rec["ahi"].append(float(hi.upper()))
    return lo, hi


R._bounds_tangent = wrapped
ubs = []
for o in cells:
    i = o["i"]; cur[0] = i
    g = np.full((10, 16), -1); g[:k] = np.unpackbits(np.frombuffer(d["hist"][i], np.uint8)).reshape(k, 16)
    lb, ub = R.mass_cell(tri, g, k, t0, budget=bud, tangent=True, zbins=24, zratio=R.fmpq(23, 20))
    ubs.append((i, lb, ub)); print(i, lb, ub, o["ub"], len(rec["t"]), flush=True)
np.savez(f"data/t412_rec_{t0}.npz", **{a: np.array(b) for a, b in rec.items()}, ubs=np.array(ubs), tri=np.array(tri), k=k)

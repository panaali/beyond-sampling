"""Cost proxy for M5t-10D on k = 10 survivors: per cell, the slice polytope's vertex count and the initial
pulling-triangulation size (no integration). usage: python tri_sizes.py SURV_FILE t0 i0 i1"""
import json
import os
import sys
import time

import numpy as np

import rigour as R

f, t0, i0, i1 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
c = np.load(f, allow_pickle=True); tri = tuple(int(v) for v in c["tri"]); k = int(c["k"]); norm = 1 / R.q(float(c["delta"]))
p1, p2, p3 = tri; out = []
for i in range(i0, min(i1, len(c["gates"]))):
    t = time.time()
    rows, _ = R.exact_rows(tri, c["gates"][i].astype(int), pick=(k == 10), order=True)
    sp = R.slice_polytope(tri, rows, NORM=norm)
    if sp is None:
        out.append(dict(i=i, empty=True)); continue
    A, b, idx = sp
    keep = R.irredundant(A, b); A = [A[j] for j in keep]; b = [b[j] for j in keep]
    P = R.Poly(A, b); v0 = P.start()
    if v0 is None or v0 == "infeasible":
        out.append(dict(i=i, empty=True)); continue
    V, inc = P.enumerate(v0, max_vertices=5000)
    S, dim = R.pulling_triangulation(V, inc, P.n)
    out.append(dict(i=i, facets=len(A), vertices=len(V), simplices=len(S), dim=dim, t=time.time() - t)); print(out[-1], flush=True)
json.dump(out, open(f"results_trisizes_{os.path.basename(f)[:-4]}_{i0}.json", "w"), indent=0)

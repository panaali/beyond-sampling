"""Benchmark nodes for T399/T400 (INTERFACE.md): (1,8,2) k = 4 cells not pruned by T390's relaxation."""
import numpy as np

import cone_count as cc
import probe
import relax
import relax2
from bnb import event_prefixes

tri = (1, 8, 2); k = 4
pre, _ = event_prefixes(tri, k, n_lines=50_000)
_, cells = cc.count(tri, k, return_cells=True)
keep = [c for c in cells if not relax.relax_infeasible(c[0], c[3], k, tri)]
mmax = max(len(c[0]) for c in keep); N = len(keep)
Gp = np.zeros((N, mmax, 10)); m = np.zeros(N, int); A = np.zeros((N, 16, 10))
rm = np.zeros(N); tm = np.zeros(N); pb = np.zeros(N); ev = np.zeros(N, int); hist = []
for i, c in enumerate(keep):
    Gp[i, :len(c[0])] = c[0]; m[i] = len(c[0]); A[i] = c[3]; hist.append(c[4])
    R = relax2.Relax(c[0], c[3], k, tri); R.forward_bounds(); rm[i] = relax2.max_margin(R)
    R.event_tighten(); tm[i] = relax2.max_margin(R)
    pb[i] = probe.probe(c[0], tri, n=2000)["best"]; ev[i] = c[4] in pre[k]
    if i % 50 == 0:
        print(i, N, rm[i], tm[i], pb[i], flush=True)
np.savez_compressed("nodes/bench_182_k4.npz", G=Gp, m=m, A=A, k=k, tri=np.array(tri), relax_margin=rm,
                    tight_margin=tm, probe_best=pb, is_event=ev, hist=np.array(hist, dtype=object), delta=relax2.DELTA)
print("saved", N)

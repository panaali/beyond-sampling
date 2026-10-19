"""T399 local benchmark (until the T398 lead's nodes/bench_182_k4.npz lands): T390's branch and bound on one triple
up to step K, saving EVERY child at each step (survivors and T390-pruned, with T390's stage) plus the event-prefix
flag, in the INTERFACE.md node format. Needs the event_cone code on PYTHONPATH (branch claude/t373-event-cone).
usage: python mkbench.py p1,p2,p3 K OUT.npz"""
import sys
import time

import numpy as np

import bnb
import cone_count as cc
import relax

tri = tuple(int(v) for v in sys.argv[1].split(",")); K = int(sys.argv[2]); out = sys.argv[3]
Wi, Wh, _ = cc.model(); H = len(Wi)
pre, ev = bnb.event_prefixes(tri, K)
G0, b0 = cc.cone(tri)
cells = [(G0, b0, cc.interior(tri), np.zeros((H, 10)), b"")]
rec = dict(G=[], A=[], k=[], stage=[], is_event=[])
for t in range(K):
    t0 = time.time(); new = bnb.step(cells, t, Wi, Wh, H); keep = []
    for c in new:
        inf, st = relax.relax_infeasible(c[0], c[3], t + 1, tri, return_stats=True)
        rec["G"].append(c[0]); rec["A"].append(c[3]); rec["k"].append(t + 1); rec["stage"].append(st["stage"] if inf else -1)
        rec["is_event"].append(c[4] in pre[t + 1])
        if not inf:
            keep.append(c)
    print(t + 1, len(new), len(keep), "%.0fs" % (time.time() - t0), flush=True)
    cells = keep
mmax = max(len(g) for g in rec["G"]); N = len(rec["G"])
G = np.zeros((N, mmax, 10)); m = np.array([len(g) for g in rec["G"]])
for i, g in enumerate(rec["G"]):
    G[i, :len(g)] = g
np.savez(out, G=G, m=m, A=np.array(rec["A"]), k=np.array(rec["k"]), tri=np.array(tri), stage=np.array(rec["stage"]),
         is_event=np.array(rec["is_event"]), event_points=ev)

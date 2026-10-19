"""T399 kill-first measurement (exact-LP reference, CPU): how many ReLU-split leaves does a benchmark node need before
the triangle relaxation (with the event rows) is infeasible in every leaf?
Per node: best-first branch and bound. A leaf = the node plus phase fixes {(step, neuron): 0/1}. In every leaf the
pre-activation bounds of all remaining steps are recomputed by exact LP (HiGHS, relax2.Relax.program with the fixes),
then the leaf is closed iff its LP with the event is infeasible. The open leaf with the largest relaxed max-margin
(relax2.max_margin) is split next, on its earliest-step unstable neuron with the largest triangle area (relax2 order).
Output per node: leaves needed to close it (or censored at BUDGET), the root margin, and the max-margin of the worst
open leaf after each split. The GPU engine can at best reproduce these LP bounds, so these curves bound what any
GPU implementation of the same relaxation + splitting could prune within a given leaf budget.
usage: python bab_cpu.py BENCH.npz OUT.json BUDGET IDX[,IDX..]"""
import json
import sys
import time

import numpy as np

import relax2

d = np.load(sys.argv[1]); out = sys.argv[2]; BUDGET = int(sys.argv[3]); idx = [int(i) for i in sys.argv[4].split(",")]
tri = tuple(int(v) for v in d["tri"]); k = int(d["k"])


class Leaf(relax2.Relax):
    def bounds_fix(self, fix):
        for s in range(self.nb):
            P, bounds = self.program(s, False, fix)
            for j in range(relax2.H):
                f = fix.get((s, j))
                r1 = relax2.solve(self.Z[s][j], P, bounds); r2 = relax2.solve(-self.Z[s][j], P, bounds)
                if r1.status == 2 or r2.status == 2:
                    return False
                lo, hi = r1.fun, -r2.fun
                if f == 1:
                    lo = max(lo, 0.)
                if f == 0:
                    hi = min(hi, 0.)
                if lo > hi:
                    return False
                self.lo[s, j], self.hi[s, j] = lo, hi
        return True


def evaluate(G, A, fix):
    """(closed, margin ub, leaf) for one leaf"""
    L = Leaf(G, A, k, tri)
    if not L.bounds_fix(fix) or not L.feasible(fix):
        return True, -np.inf, L
    return False, relax2.max_margin(L, fix), L


def pick(L, fix):
    c = [(s, -(u * -l) / (u - l), j) for s in range(L.nb) for j in range(relax2.H)
         for l, u in [(L.lo[s, j], L.hi[s, j])] if l < 0 < u and (s, j) not in fix]
    return min(c)[0::2] if c else None


res = []
for i in idx:
    G = d["G"][i][:int(d["m"][i])]; A = d["A"][i]; t0 = time.time()
    closed, mg, L = evaluate(G, A, {})
    rec = dict(idx=i, is_event=bool(d["is_event"][i]), probe_best=float(d["probe_best"][i]), root_margin=float(mg),
               relax_margin=float(d["relax_margin"][i]), curve=[], leaves=1, closed=closed)
    opened = [] if closed else [(-mg, 0, {}, L)]; nleaves = 1; cnt = 1
    while opened and nleaves < BUDGET:
        opened.sort(key=lambda a: a[0]); negm, _, fix, L = opened.pop(0)
        sj = pick(L, fix)
        if sj is None:                                                   # no unstable neuron left: LP exact, stays open
            opened.append((negm, cnt, fix, L)); break
        nleaves += 1
        for f in (1, 0):
            cl, m2, L2 = evaluate(G, A, {**fix, sj: f}); cnt += 1
            if not cl:
                opened.append((-m2, cnt, {**fix, sj: f}, L2))
        rec["curve"].append((nleaves, len(opened), float(-min(a[0] for a in opened)) if opened else None))
    rec.update(leaves=nleaves, closed=not opened, open_left=len(opened), sec=time.time() - t0)
    res.append(rec); print(i, {a: rec[a] for a in ("is_event", "probe_best", "root_margin", "leaves", "closed", "open_left", "sec")}, flush=True)
    json.dump(dict(budget=BUDGET, tri=tri, k=k, nodes=res), open(out, "w"), indent=1)

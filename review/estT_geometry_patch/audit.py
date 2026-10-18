"""T316 V3 / locator: walk subtrees (no full masses); for each leaf compare estT's stored ray count with an independent qhull
vertex enumeration; for every cell where they differ, compute its mass both ways (orig pulling triangulation vs patch) and sum
the change. usage: T270_K=6 python audit.py SRC POS[,POS...] -> audit_k{K}.json (incremental)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.abspath(sys.argv[1]); sys.path[:0] = [HERE, SRC]
os.chdir(SRC)
import numpy as np
import prefix6 as Q, prefix as P, gmass, estG  # noqa: E402
import estT_base as B  # noqa: E402
from m43 import Ops  # noqa: E402
import estT_geometry as G  # noqa: E402
model, front = Q.front_list(11); P._init(model, Q.K, True); ORIG = B.mass
fn = os.path.join(HERE, f"audit_k{Q.K}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}


def cell_mass(c, patched):
    B.mass = estG._tri_mass = ORIG
    if patched:
        G.apply_estG(estG, B)
    col = P._Col(); estG.BUF[0] = col; B.mass(c, Ops()); B.mass = estG._tri_mass = ORIG
    return float(gmass.grouped(np.array(col.R).reshape(-1, Q.K, Q.K), np.zeros(len(col.R), np.int64), 1, "cpu")[0]) if col.R else 0.0


for p in map(int, sys.argv[2].split(",")):
    if str(p) in out:
        continue
    t0 = time.time(); stack = [front[p]]; leaves = nonsimp = defect = fail = 0; dm = 0.0; worst = []
    while stack:
        c, g, A, t, hist = stack.pop()
        if t == Q.K:
            leaves += 1
            if len(c.R) == Q.K:
                continue
            nonsimp += 1
            try:
                S = G.simplices(c)
            except Exception:  # noqa: BLE001
                S = None
            if S is None:
                fail += 1; continue
            nq = len(np.unique(np.round(S.reshape(-1, Q.K), 10), axis=0))
            if nq != len(c.R):
                defect += 1; mo, mp = cell_mass(c, False), cell_mass(c, True); dm += mp - mo
                worst.append((mp - mo, len(c.R), nq))
            continue
        for c2, g2, A2 in P._children(c, g, A, t, Ops()):
            stack.append((c2, g2, A2, t + 1, hist + np.packbits(g2 > 0).tobytes()))
    worst.sort(reverse=True)
    out[str(p)] = dict(leaves=leaves, nonsimplicial=nonsimp, ray_count_mismatch=defect, qhull_fail=fail, mass_change=dm,
                       worst=worst[:5], sec=time.time() - t0)
    print(p, json.dumps(out[str(p)]), flush=True); json.dump(out, open(fn, "w"), indent=1)

"""T131 diag: exact-mode on zoo(8,4,seed); record every leaf cell and every 4-D simplex mass; compare the
fixed Plackett quadrature against a much finer rule. -> diag1_s{seed}.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro")]
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo
sd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
m = zoo(8, 4, sd)
SIMP = []; orig = estT.orthant
def rec(R, ops):
    v = orig(R, ops); SIMP.append((R.copy(), v)); return v
estT.orthant = rec
LEAF = [0.0]; origce = estT._cell_error
def ce(c, g, A, Wo, truth, ops):
    n0 = len(SIMP); LEAF[0] += estT.mass(c, Ops()); del SIMP[n0:]
    return origce(c, g, A, Wo, truth, ops)
estT._cell_error = ce
t0 = time.time(); acc = estT.estT(m, Ops(), mode="exact")
print(f"seed {sd} acc {acc:.10f} leafsum {LEAF[0]:.10f} err-simplices {len(SIMP)} {time.time()-t0:.0f}s")
estT.orthant = orig
fine = estT._rule(400, 3)
def orth_rule(R, rule):
    old = dict(estT._RULES); estT._RULES["thin"] = rule; estT._RULES["wide"] = rule
    v = orig(R, Ops()); estT._RULES.update(old); return v
d = np.array([v - orth_rule(R, fine) for R, v in SIMP])
neg = sum(v < 0 for _, v in SIMP)
print(f"err-simplex diff (fixed - fine400): sum {d.sum():.3e} max|.| {np.abs(d).max():.3e} neg masses {neg}")
np.save(os.path.join(HERE, f"simp_s{sd}.npy"), np.array([R for R, _ in SIMP]))
json.dump(dict(acc=acc, leafsum=LEAF[0], n=len(SIMP), dsum=d.sum(), dmax=np.abs(d).max(), neg=int(neg)),
          open(os.path.join(HERE, f"diag1_s{sd}.json"), "w"))

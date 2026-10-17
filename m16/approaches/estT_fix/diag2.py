"""T131 residual: zoo(8,4,seed) exact mode; per half-chamber leaf-mass deficit and the negative simplex masses."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro")]
import numpy as np, estT
from m43 import Ops
from zoo44 import zoo
sd = int(sys.argv[1]); m = zoo(8, 4, sd); V = []; o0 = estT.orthant
def rec(R, ops):
    v = o0(R, ops); V.append((abs(np.linalg.det(R)), v)); return v
estT.orthant = rec; LEAF = [0.0]
def ce(c, g, A, Wo, truth, ops):
    LEAF[0] += estT.mass(c, Ops()); return 0.0
estT._cell_error = ce; estT.MODE[0] = "exact"; rows = []
for h in estT.halves(4):
    LEAF[0] = 0.0; hm = estT.mass(h["cone"], Ops()); g1 = (h["sigma"] * m[0] > 0).astype(float)
    estT._finish(h["cone"], g1, np.outer(m[0], np.eye(4)[0]), 1, m, h["truth"], h["n_true"], 0.1, Ops(), True)
    rows.append(LEAF[0] - hm)
rows = np.array(rows); V = np.array(V)
print("total deficit %.3e; worst halves %s" % (rows.sum(), np.sort(rows)[:4]))
neg = V[:, 1] < 0
print("simplices %d, neg %d sum %.3e; det<1e-9: %d sum %.3e; det<1e-6 sum %.3e" % (len(V), neg.sum(), V[neg, 1].sum(),
      (V[:, 0] < 1e-9).sum(), V[V[:, 0] < 1e-9, 1].sum(), V[V[:, 0] < 1e-6, 1].sum()))
for lo, hi in [(0, 1e-12), (1e-12, 1e-10), (1e-10, 1e-8), (1e-8, 1e-6)]:
    s = (V[:, 0] >= lo) & (V[:, 0] < hi); print(f"det [{lo:.0e},{hi:.0e}) n {s.sum()} sum {V[s,1].sum():.3e} min {V[s,1].min() if s.any() else 0:.2e}")

"""Diagnostic (uses random inputs only to measure TRUE cone masses; not part of the estimator).
Where does the harmonic estimate lose mass?  Compares each leaf's propagated mass c_0 with its exact mass
(Monte Carlo over the leaf's cone), grouped by readout outcome (all-correct / all-wrong / split).
usage: python diag_leaves.py T D model"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from leaderboard import models
import hprop

T, D, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
m = models(16, T)[name]; Wi, Wh, Wo = m
# re-run propagation, keeping leaves: monkeypatch readout by capturing br
cap = {}
orig = hprop.readout_global
def grab(sp, br, Wo_, T_):
    cap["br"] = br; return 0.0
hprop.readout_global = grab
hprop.estimate(m, D, cells="global")
br = cap["br"]
x = np.random.default_rng(5).standard_normal((2_000_000, T))
# leaf membership of each x: all constraints of leaf cone (use rays? use gates): recompute gates and compare
h = np.zeros((len(x), 16)); keys = []
G = []
for t in range(T):
    z = h @ Wh.T + x[:, t:t+1] * Wi[None]; g = z > 0; h = np.where(g, z, 0); G.append(g)
o = np.argsort(x, 1)
correct = (h @ Wo.T).argmax(1) == o[:, -2]
# map leaf by its final A: A_b x == h for points in the leaf
res = []
sp = hprop.space(T, D)
for (_, K, c, A) in br:
    L = Wo @ A
    # leaf indicator: points whose h equals A x
    inn = np.all(np.abs(x @ A.T - h) < 1e-9 * (1 + np.abs(h)), 1)
    # within leaf, the true correct fraction is determined by cell & logits
    res.append((c[0], inn.mean(), (inn & correct).mean(), (inn & ~correct).mean()))
res = np.array(res)
print("sum c0", res[:, 0].sum(), "sum true", res[:, 1].sum())
err_leaf = res[:, 3] > 0
print("leaves", len(res), "with errors", err_leaf.sum())
print("mass on leaves containing errors: propagated %.5f true %.5f (true err inside %.5f)" %
      (res[err_leaf, 0].sum(), res[err_leaf, 1].sum(), res[:, 3].sum()))
small = res[:, 1] < 1e-3
print("leaves with true mass <1e-3:", small.sum(), "propagated %.5f true %.5f" % (res[small, 0].sum(), res[small, 1].sum()))
print("abs mass misallocation sum |c0-true| = %.4f" % np.abs(res[:, 0] - res[:, 1]).sum())

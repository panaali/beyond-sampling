"""Where does the variance of the CMC value Y = P(correct | x_0..x_8) live? (original model)"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../.."), HERE]
from common import Wi, Wh, Wo
from core import init, step, last_step_value
M = (Wi, Wh, Wo); n = 100_000; rng = np.random.default_rng(3)
X = rng.standard_normal((n, 9)); S = init(n)
for t in range(9): step(M, S, X[:, t])
y = last_step_value(M, S); d2 = (y - y.mean()) ** 2; tot = d2.sum()
s = -np.sort(-X, 1); g12 = (s[:, 0] - s[:, 1]) / np.abs(s[:, 0]); g23 = (s[:, 1] - s[:, 2]) / np.maximum(np.abs(s[:, 1]), 1e-9)
blind = s[:, 1] <= 0.3
print(f"Var(Y)={y.var():.4f}; share of Var from Y<0.9: {d2[y<0.9].sum()/tot:.2f} (prob {np.mean(y<0.9):.3f})")
for name, m in [("top-2 of x0..x8 near-tie g12<0.05", g12 < 0.05), ("2nd/3rd near-tie g23<0.05", g23 < 0.05),
                ("2nd largest of x0..x8 <= 0.3", blind)]:
    print(f"  {name:36s}: prob {m.mean():.3f}, share of Var {d2[m].sum()/tot:.2f}")
# R^2 of Y on functions of (m1, m2) only (binned 2-D means) and on (m1,m2,p1,p2)
def r2(keys):
    _, inv = np.unique(keys, axis=0, return_inverse=True); inv = inv.ravel()
    mu = np.bincount(inv, y) / np.bincount(inv); return 1 - ((y - mu[inv]) ** 2).mean() / y.var()
b = lambda v, k=40: np.clip(((v + 3) / 6 * k).astype(int), 0, k)
print(f"R^2 of Y on binned (m1,m2): {r2(np.c_[b(S['m1']), b(S['m2'])]):.3f}; + (p1,p2): {r2(np.c_[b(S['m1'],20), b(S['m2'],20), S['p1'], S['p2']]):.3f} (in-sample, optimistic)")

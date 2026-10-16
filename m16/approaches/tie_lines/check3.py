"""Swap (permutation) conditional MC: average g over measure-preserving position swaps of top ranks."""
import sys, itertools
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tie_lines")
from common import Wi, Wh, Wo
from tielines import correct
M = (Wi, Wh, Wo)
N = 300000
x = np.random.default_rng(3).standard_normal((N, 10))
o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1); r = np.arange(N)
def perm(x, a, b):  # swap positions of ranks a,b
    y = x.copy(); y[r, o[:, a]] = x[r, o[:, b]]; y[r, o[:, b]] = x[r, o[:, a]]; return y
G = {}
for p in itertools.permutations(range(3)):
    y = x.copy()
    for k in range(3): y[r, o[:, k]] = x[r, o[:, p[k]]]
    G[p] = correct(M, y).astype(float)
g = G[(0, 1, 2)]; vg = g.var()
s23 = G[(0, 2, 1)]; s12 = G[(1, 0, 2)]
S3 = np.mean([G[p] for p in G], 0)
print("plain var %.5f" % vg)
g23 = (xs[:, 1] - xs[:, 2]) / np.maximum(np.abs(xs[:, 1]), 1e-9)
g12 = (xs[:, 0] - xs[:, 1]) / np.abs(xs[:, 0])
blind = xs[:, 1] <= 0
for name, est_all, k in [("swap23", (g + s23) / 2, 2), ("swap12", (g + s12) / 2, 2), ("S3", S3, 6),
                         ("swap23+12", (g + s23 + s12) / 3, 3)]:
    print(name, "all: var %.5f eff %.2f" % (est_all.var(), vg / (est_all.var() * k)))
    for c in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        gg = np.minimum(g23, g12) if name in ("S3", "swap23+12") else (g23 if name == "swap23" else g12)
        sel = (gg < c) & ~blind
        e = np.where(sel, est_all, g); cost = 1 + (k - 1) * sel.mean()
        print("   c=%.2f sel %.3f mean %.5f var %.5f cost %.3f eff %.2f" % (c, sel.mean(), e.mean(), e.var(), cost, vg / (e.var() * cost)))
# combined rule: swap23 if g23<c23, swap12 if g12<c12 (both -> 3 evals)
for c23, c12 in [(0.05, 0.05), (0.1, 0.1), (0.1, 0.05), (0.2, 0.1)]:
    a = (g23 < c23) & ~blind; b = (g12 < c12) & ~blind
    e = np.where(a & b, (g + s23 + s12) / 3, np.where(a, (g + s23) / 2, np.where(b, (g + s12) / 2, g)))
    cost = 1 + a.mean() + b.mean()
    print("combo", c23, c12, "var %.5f cost %.3f eff %.2f" % (e.var(), cost, vg / (e.var() * cost)))

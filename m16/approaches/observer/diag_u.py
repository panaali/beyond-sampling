"""DIAGNOSTIC: predicted vs sampled error by relative gap (u/b for 23 flips, g/b for 12 flips)."""
import sys, numpy as np
sys.path.insert(0, '/home/user/clean-neurons/m16')
from v2 import estimate
from quad import grid
from arc_eval import forward
from common import Wi, Wh, Wo
q = int(sys.argv[1]) if len(sys.argv) > 1 else 2
o, _ = estimate((Wi, Wh, Wo), 3, 3, q)
C, U, G, W = grid(20, 20, 20); B = C + U; A = B + G
x = np.random.default_rng(3).standard_normal((1_000_000, 10)); o_ = np.argsort(x, 1)
out = forward((Wi, Wh, Wo), x).argmax(1); s = np.sort(x, 1)
b = s[:, -2]; pos = b > 0
edges = [0, .01, .02, .05, .1, .2, .4, 1, 100]
for kind, gp, gs, wrong in (('23', U / B, (s[:, -2] - s[:, -3]) / b, o_[:, -3]), ('12', G / B, (s[:, -1] - s[:, -2]) / b, o_[:, -1])):
    print(kind)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gp >= lo) & (gp < hi); ms = pos & (gs >= lo) & (gs < hi)
        print(f'  rel gap [{lo},{hi}): pred {o[kind + "_node"][m].sum():.5f}  samp {(ms & (out == wrong)).mean():.5f}')

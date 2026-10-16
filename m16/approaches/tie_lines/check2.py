import sys, time
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tie_lines")
from common import Wi, Wh, Wo
from tielines import hybrid_terms, correct
M = (Wi, Wh, Wo)
x = np.random.default_rng(2).standard_normal((40000, 10))
g = correct(M, x); vg = g.var()
print("plain mean %.5f var %.5f" % (g.mean(), vg))
for c23, c12 in [(0.02, 0.02), (0.05, 0.05), (0.1, 0.1), (0.05, 0.0), (0.2, 0.1)]:
    t0 = time.time()
    v, c, a, b = hybrid_terms(M, x, c23, c12)
    print("c23 %.2f c12 %.2f mean %.5f var %.5f cost %.3f | sel %.3f %.3f pieces/line %.1f %.1f | eff gain %.2f | %.0fs" % (
        c23, c12, v.mean(), v.var(), c.mean(), a.mean(), b.mean(), c[a].mean() / 2, c[b].mean() / 2 if b.any() else 0,
        vg / (v.var() * c.mean()), time.time() - t0))

import sys, time
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tie_lines")
from common import Wi, Wh, Wo
from tielines import Z_lines, correct, pair_lines
M = (Wi, Wh, Wo)
rng = np.random.default_rng(1)
# validate one line by brute force
x = rng.standard_normal((5, 10))
for kind in ("23", "12"):
    Z, p = Z_lines(M, x, kind)
    base, d, tau, tp, tn, m, xs = pair_lines(x, kind)
    for n in range(3):
        t = rng.standard_normal(2_000_000); t = t[np.abs(t) < tau[n]][:400000]
        xx = base[n] + t[:, None] * d[n]
        print(kind, Z[n], correct(M, xx).mean(), p[n], tau[n])
for kind in ("23", "12"):
    x = rng.standard_normal((20000, 10)); t0 = time.time()
    Z, p = Z_lines(M, x, kind); g = correct(M, x)
    print(kind, "meanZ %.5f se %.5f | var Z %.5f var g %.5f | pieces mean %.2f | %.1fs" % (
        Z.mean(), Z.std() / np.sqrt(len(Z)), Z.var(), g.var(), p.mean(), time.time() - t0))

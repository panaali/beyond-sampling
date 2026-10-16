import sys, time
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tie_lines")
from common import Wi, Wh, Wo
from tielines import Z_stratum, correct
from stratified import stratum, draw, EDGES
M = (Wi, Wh, Wo)
rng = np.random.default_rng(5)
for hst in [1, 2, 3, 4]:
    x = draw(hst, 3000, rng); t0 = time.time()
    Z, p = Z_stratum(M, x, EDGES); g = correct(M, x)
    print("stratum %d: mean g %.4f Z %.4f | sd g %.3f Z %.3f | pieces %.2f | sd*sqrt(2p) %.3f sd*sqrt(p) %.3f | %.0fs" % (
        hst, g.mean(), Z.mean(), g.std(), Z.std(), p.mean(), Z.std() * np.sqrt(2 * p.mean()), Z.std() * np.sqrt(p.mean()), time.time() - t0))

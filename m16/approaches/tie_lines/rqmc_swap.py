"""Does RQMC help once the tie discontinuity is symmetrized (swap) ? Original model, scrambled Sobol."""
import sys
import numpy as np
from scipy.stats import qmc, norm
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tie_lines")
from common import Wi, Wh, Wo
from tielines import correct
M = (Wi, Wh, Wo)
def swapmin(x):
    N = len(x); o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1); r = np.arange(N)
    g23 = (xs[:, 1] - xs[:, 2]) / np.maximum(np.abs(xs[:, 1]), 1e-12); g12 = (xs[:, 0] - xs[:, 1]) / np.abs(xs[:, 0])
    a = np.where(g23 < g12, 1, 0); b = a + 1
    y = x.copy(); y[r, o[r, a]] = x[r, o[r, b]]; y[r, o[r, b]] = x[r, o[r, a]]
    return (correct(M, x).astype(float) + correct(M, y)) / 2
if __name__ == '__main__':
  for n in [2048, 8192]:
    q, m = [], []
    for s in range(24):
        u = qmc.Sobol(10, scramble=True, seed=s).random(n)
        q.append(swapmin(norm.ppf(u)).mean())
        m.append(swapmin(np.random.default_rng(100 + s).standard_normal((n, 10))).mean())
    print(n, "sd RQMC %.5f  sd iid %.5f  var ratio %.2f" % (np.std(q), np.std(m), np.var(m) / np.var(q)))

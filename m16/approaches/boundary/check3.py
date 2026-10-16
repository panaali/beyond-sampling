import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from tie23 import segments_23
from local import local_band
from scipy.special import ndtr
M = (Wi, Wh, Wo)
rng = np.random.default_rng(1)
x = rng.standard_normal((40000, 10))
x0, d, lo, hi, tn, tp, yb = segments_23(x)
band, sr, sl, p = local_band(M, x0, d, hi, -lo, tp, tn)
tot = ndtr(hi) - ndtr(lo); fb = band / tot
s = np.sort(x, 1)
for name, m in [("ybar<=0", yb <= 0), ("ybar>0", yb > 0)]:
    print(name, "P", m.mean(), "mean", fb[m].mean(), "sd", fb[m].std(), "contrib E[f^2]", (fb**2 * m).mean())
print("total var", fb.var())
# by segment length
g = (s[:, -2] - s[:, -3])
for q in [0, .05, .1, .2, .4, 1, 9]:
    pass
qs = np.quantile(hi, [0, .1, .25, .5, .75, 1])
for i in range(5):
    m = (hi >= qs[i]) & (hi < qs[i+1]) & (yb > 0)
    print("T bin", qs[i].round(3), "mean", fb[m].mean().round(4), "sd", fb[m].std().round(4))
print("frac with band mass==0", (band == 0).mean())
print("width (sr+sl) quantiles", np.quantile(sr + sl, [.1, .5, .9, .99]))

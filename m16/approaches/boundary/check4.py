import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from arc_eval import forward
from tie23 import segments_23
from local import local_band
M = (Wi, Wh, Wo)
rng = np.random.default_rng(2)
x = rng.standard_normal((100000, 10))
x0, d, lo, hi, tn, tp, yb = segments_23(x)
band, sr, sl, p = local_band(M, x0, d, hi, -lo, tp, tn)
t0 = np.einsum("ij,ij->i", x - x0, d)
inband = (t0 <= sr) & (t0 >= -sl)
out = forward(M, x).argmax(1)
o = np.argsort(-x, 1); rank = np.argsort(o, 1)[np.arange(len(x)), out]
wrong = rank != 1
rem = wrong & ~inband
print("wrong", wrong.mean(), "in band", (wrong & inband).mean(), "rem", rem.mean())
s = -np.sort(-x, 1)
blind = s[:, 1] <= 0
for r in range(10):
    print("rem picks rank", r + 1, (rem & (rank == r)).mean(), " of which blind", (rem & (rank == r) & blind).mean())
gap12 = (s[:, 0] - s[:, 1]); 
m = rem & (rank == 0) & ~blind
print("rank1 rem: gap12 quantiles", np.quantile(gap12[m], [.5, .9]), "overall", np.quantile(gap12, [.5]))

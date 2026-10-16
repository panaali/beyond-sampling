import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from bands import per_sample, point_out
M = (Wi, Wh, Wo)
x = np.random.default_rng(4).standard_normal((60000, 10))
r = per_sample(M, x)
out = point_out(M, x); o = np.argsort(-x, 1); rank = np.argsort(o, 1)[np.arange(len(x)), out]
s = -np.sort(-x, 1); R = r["R"] > 0
print("R", R.mean(), "passes", r["passes"].mean())
for k in range(6): print(" rank", k + 1, (R & (rank == k)).mean(), "blind", (R & (rank == k) & (s[:, 1] <= 0)).mean())
m = R & (rank == 2)
print("rank3 R: gap23", np.quantile((s[:,1]-s[:,2])[m], [.1,.5,.9]), "gap34", np.quantile((s[:,2]-s[:,3])[m], [.1,.5,.9]), "gap12", np.quantile((s[:,0]-s[:,1])[m], [.1,.5,.9]))
m = R & (rank == 0)
print("rank1 R: gap12", np.quantile((s[:,0]-s[:,1])[m], [.1,.5,.9]), "s2", np.quantile(s[m,1], [.1,.5,.9]))

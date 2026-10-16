import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from bands import per_sample, point_out
M = (Wi, Wh, Wo)
seed = int(sys.argv[1])
x = np.random.default_rng(seed).standard_normal((100000, 10))
r = per_sample(M, x)
g = r["fA"] + r["fB"] + r["R"]
ind = (point_out(M, x) != np.argsort(x, 1)[:, -2]) & (np.sort(x, 1)[:, -2] > 0)
dd = g - ind
print(seed, "g", g.mean(), "ind", ind.mean(), "diff", dd.mean(), "+-", dd.std() / np.sqrt(len(dd)))

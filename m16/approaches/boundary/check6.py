import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from bands import per_sample, point_out
M = (Wi, Wh, Wo)
x = np.random.default_rng(6).standard_normal((60000, 10))
t = time.time(); r = per_sample(M, x); print("time", time.time() - t)
g = r["fA"] + r["fB"] + r["R"]
xb = np.random.default_rng(7).standard_normal((3000000, 10)); xb = xb[np.sort(xb, 1)[:, -2] <= 0][:20000]
wb = (point_out(M, xb) != np.argsort(xb, 1)[:, -2]).mean()
P = 11 / 1024
print("nonblind err", g.mean(), "+-", g.std() / np.sqrt(len(g)), "blind err rate", wb)
print("acc est", 1 - g.mean() - P * wb, "truth 0.953179")
for k in ["fA", "fB", "R"]: print(k, r[k].mean(), r[k].var())
print("var g", g.var(), "passes", r["passes"].mean(), "var*cost", g.var() * r["passes"].mean())

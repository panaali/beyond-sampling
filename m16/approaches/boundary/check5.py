import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from bands import per_sample
M = (Wi, Wh, Wo)
x = np.random.default_rng(3).standard_normal((40000, 10))
t = time.time(); r = per_sample(M, x); print("time", time.time() - t)
e = r["fA"] + r["fB"] + r["R"]
print("acc est", 1 - e.mean(), "+-", e.std() / np.sqrt(len(e)))
for k in ["fA", "fB", "R"]:
    print(k, "mean", r[k].mean(), "var", r[k].var())
print("var total", e.var(), "passes", r["passes"].mean())
print("gain", 0.0447 / (e.var() * r["passes"].mean()))

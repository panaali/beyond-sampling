import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from arc_eval import forward
from common import Wi, Wh, Wo
from tie23 import cond_acc_23
M = (Wi, Wh, Wo)
rng = np.random.default_rng(1)
x = rng.standard_normal((20000, 10))
t = time.time(); f, p = cond_acc_23(M, x); print("time", time.time() - t)
ind = (forward(M, x).argmax(1) == np.argsort(x, 1)[:, -2])
print("mean f", f.mean(), "+-", f.std() / np.sqrt(len(f)), " indicator", ind.mean())
print("var f", f.var(), "var ind", ind.var(), "pieces mean", p.mean(), "max", p.max())
print("var*cost", f.var() * p.mean(), "ratio vs sampling", ind.var() / (f.var() * p.mean()))
print("frac f<1", (f < 1 - 1e-12).mean())

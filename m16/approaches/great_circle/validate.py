import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/great_circle")
from common import Wi, Wh, Wo
from arc_eval import forward
from gcirc import walk, random_planes
M = (Wi, Wh, Wo)
rng = np.random.default_rng(1)
U, V = random_planes(20, rng)
t = time.time(); f, p, pc = walk(M, U, V); print("time", time.time() - t)
th = (np.arange(200000) + 0.5) / 200000 * 2 * np.pi
for i in range(20):
    x = np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i]
    g = (forward(M, x).argmax(1) == np.argsort(x, 1)[:, -2]).mean()
    print(i, f"{f[i]:.6f} grid {g:.6f} pieces {p[i]} partial {pc[i]:.1f}")

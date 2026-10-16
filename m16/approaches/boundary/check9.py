import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
import zones2 as Z2
M = (Wi, Wh, Wo)
rng = np.random.default_rng(1)
gA, gB = float(sys.argv[1]), float(sys.argv[2])
t = time.time()
G = Z2.geom_A(M, 30000, rng, gA); v, p = Z2.term(M, G, rng)
print("A2", v.mean(), "+-", v.std() / np.sqrt(len(v)), "pcs", p.mean(), "s", np.sqrt(v.var() * p.mean()), time.time() - t, flush=True)
G = Z2.geom_B(M, 30000, rng, gA, gB); v, p = Z2.term(M, G, rng)
print("B2", v.mean(), "+-", v.std() / np.sqrt(len(v)), "pcs", p.mean(), "s", np.sqrt(v.var() * p.mean()))

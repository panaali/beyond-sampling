import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
import zones as Z
M = (Wi, Wh, Wo)
print("P_ACC", Z.P_ACC_A, Z.P_ACC_B)
rng = np.random.default_rng(0)
# validate zone terms vs indicator on x samples (same zones)
x = rng.standard_normal((2_000_000, 10))
zz = Z.zone_of(x, *map(float, sys.argv[1:3]))
gA, gB = map(float, sys.argv[1:3])
w = np.zeros(len(x), bool)
for k in range(0, len(x), 500000): w[k:k+500000] = Z.point_wrong(M, x[k:k+500000])
for zn in range(4): print("zone", zn, "P", (zz == zn).mean(), "err mass", (w & (zz == zn)).mean(), "rate", w[zz == zn].mean())
t = time.time()
vA, pA = Z.term_A(M, 40000, rng, gA); print("A", vA.mean(), "+-", vA.std() / 200, "pieces", pA.mean(), "2nd moment", (vA**2).mean(), "var", vA.var(), time.time() - t)
vB, pB = Z.term_B(M, 40000, rng, gA, gB); print("B", vB.mean(), "+-", vB.std() / 200, "pieces", pB.mean(), "var", vB.var())
for name, v, p in [("A", vA, pA), ("B", vB, pB)]:
    print(name, "sigma*sqrt(cost)", np.sqrt(v.var() * p.mean()))

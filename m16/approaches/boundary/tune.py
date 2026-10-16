import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
import zones as Z
M = (Wi, Wh, Wo)
rng = np.random.default_rng(0)
x = rng.standard_normal((1_000_000, 10))
w = Z.point_wrong(M, x)
for gA in [0.05, 0.2, 0.4]:
    for gB in [0.05, 0.2, 0.4]:
        zz = Z.zone_of(x, gA, gB)
        PR = (zz == 3).mean(); rR = w[zz == 3].mean(); PB = (zz == 0).mean(); rb = w[zz == 0].mean()
        vA, pA = Z.term_A(M, 12000, rng, gA)
        vB, pB = Z.term_B(M, 12000, rng, gA, gB)
        sA = np.sqrt(vA.var() * pA.mean()); sB = np.sqrt(vB.var() * pB.mean())
        sR = PR * np.sqrt(rR * (1 - rR)); sb = PB * np.sqrt(rb * (1 - rb))
        tot = sA + sB + sR + sb
        print(f"gA {gA} gB {gB} | A mean {vA.mean():.4f} pcs {pA.mean():.2f} s {sA:.4f} | B {vB.mean():.4f} pcs {pB.mean():.2f} s {sB:.4f} | R P {PR:.3f} rate {rR:.4f} s {sR:.4f} | blind s {sb:.4f} | total {tot:.4f} gain {0.0446/tot**2:.2f}", flush=True)

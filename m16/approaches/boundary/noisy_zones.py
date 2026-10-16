import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from arc_eval import noisy_models
import zones as Z
gA, gB = 0.15, 0.1
x = np.random.default_rng(0).standard_normal((600000, 10))
zz = Z.zone_of(x, gA, gB)
for sig in [0.0025, 0.005]:
    for mi, m in enumerate(noisy_models(6, sig)):
        w = Z.point_wrong(m, x)
        print(sig, mi, "err %.4f" % w.mean(), " ".join("z%d P%.3f rate %.4f mass %.4f" % (k, (zz == k).mean(), w[zz == k].mean(), (w & (zz == k)).mean()) for k in range(4)), flush=True)

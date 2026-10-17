"""T75: the 'crossover slope ~ 1' failure window found by the T74 review (scan one weight of the clean seed-0 model).
usage: python window75.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from m43 import Ops, Wi0, Wh0, Wo0
from exact import exact_acc
import est_mech as EM
from est75 import est75, RECIPE

for (r, c), deltas in (((0, 3), np.arange(-0.40, 0.01, 0.05)), ((2, 3), np.arange(0.0, 0.81, 0.1))):
    print(f"scan Wh[{r},{c}]:")
    for d in deltas:
        Wh = Wh0.copy(); Wh[r, c] += d; m = (Wi0, Wh, Wo0)
        t = exact_acc(m); eb = EM.est_mech(m) - t
        o = Ops(); e = est75(m, o, **RECIPE) - t
        print(f"  delta {d:+.2f}: acc {t:.3f}  T63 tier B err {100*eb:+.2f} pp | T75 recipe err {100*e:+.2f} pp ({o.n} ops)")

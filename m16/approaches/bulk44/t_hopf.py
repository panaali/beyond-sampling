import sys, os, time; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
from zoo44 import zoo; from m43 import Ops; import hopf
m = zoo(4, 4, 0)   # run87 exact 0.47691
for nu in (4, 8, 16, 32):
    o = Ops(); t0 = time.time(); e = hopf.est_hopf(m, nu, o); print(nu, round(e, 6), o.n, f"{time.time()-t0:.1f}s")

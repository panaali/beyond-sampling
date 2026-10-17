"""T87 fix attempts on zoo (4,4) seed 0 (clean): F4 = intermediate steps resolved exactly + margin rule at the last
step, with a cheaper orthant rule (6/16 Gauss nodes); F6 = full enumeration (exact mode) with the cheaper rule.
usage: python diag87c.py -> diag87c.log"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo

m = zoo(4, 4, 0); ex = 0.4769097   # exact (12/48-node rule), run87.json
p = ex; OPS_S = 96
estT._RULES = {"wide": estT._rule(6, 2), "thin": estT._rule(16, 3)}
orig = estT._split_impact
estT._split_impact = lambda cells, A, w, eps, ops, exact: orig(cells, A, w, eps, ops, True)
o = Ops(); t0 = time.time(); e = estT.estT(m, o, eps=0.1)
r = (e - ex) ** 2 / (p * (1 - p) / (o.n / OPS_S))
print(f"F4 intermediate exact + margin last, cheap orthant: err {100*(e-ex):+.3f}pp ops {o.n/1e6:.1f}M ratio {r:.2f} ({time.time()-t0:.0f}s)", flush=True)
estT._split_impact = orig
o = Ops(); t0 = time.time(); e = estT.estT(m, o, mode="exact")
r = (e - ex) ** 2 / (p * (1 - p) / (o.n / OPS_S))
print(f"F6 full enumeration, cheap orthant: err {100*(e-ex):+.5f}pp ops {o.n/1e6:.1f}M ratio {r:.4f} ({time.time()-t0:.0f}s)", flush=True)

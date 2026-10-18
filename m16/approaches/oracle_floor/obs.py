"""E21 observer (observer/est.py, v4 = cone-clipped noise, nq=12, noise=1) on DEV2. DEDUCTIVE: no random model
inputs (the LHS bin assignment uses a fixed seed and touches no model; quadrature, tie levels, bin moments are
model-free). est.py's `make` is used unchanged for the estimate; the cost is re-instrumented here:
  per anchor (core.map_and_cone row + v4 per-anchor work), counted op by op, H=16, T=10:
    per step: z = h Wh^T + x Wi       2H^2 + 2H        =   544
              P = Wh J (+Wi)         2H^2 T + H       = 5,136
              delta = -z/P           2HT              =   320
              pos/neg min/max+where  4HT              =   640
              lo/hi update           4T               =    40
              h, J gating            H + HT           =   176      -> 6,856 x 10 steps = 68,560
    readout h Wo^T + Wo J            2*10*16 + 2*10*16*10            =  3,520
    v4 per anchor: D row, clip+mean+var of 7 rest x 24 nodes, mu/sd sums  ~   720
    => 72,800 FLOPs per anchor (12.24 FPE; est.py charges 10.5)
  quadrature: per (kind, triple, LHS row, node): 6 (top-3 terms) + 1 + 1 div + ndtr (counted 30) + 3 = 41 FLOPs
  blind probe: 10 forward passes.
FPE = 5948 FLOPs.  usage: python obs.py nv ng q [first] [last]"""
import json
import os
import sys
import time

import numpy as np

from panel import HERE, ROOT, dev2, target
import common  # noqa: F401  (shim, so arc_eval imports without torch)
sys.path.insert(0, os.path.join(ROOT, "m16/approaches/observer"))
import est as E  # noqa: E402
import v4  # noqa: E402
from quad import grid  # noqa: E402

FPE = 5948
ANCHOR_FLOPS = 72_800
NQ = 12
nv, ng, q = (int(a) for a in sys.argv[1:4])
models = [("target", target())] + dev2()
lo = int(sys.argv[4]) if len(sys.argv) > 4 else 0
hi = int(sys.argv[5]) if len(sys.argv) > 5 else len(models)
OUT = os.path.join(HERE, f"results/obs_{nv}{ng}{q}"); os.makedirs(OUT, exist_ok=True)

_count = {"anchors": 0}
_mac = v4.map_and_cone
def _counted(model, x):
    _count["anchors"] += len(x)
    return _mac(model, x)
v4.map_and_cone = _counted

f = E.make(nv=nv, ng=ng, q=q, nq=NQ, noise=1.0, ver=4)
nodes = len(grid(NQ, NQ, NQ)[0])
for k in range(lo, hi):
    name, m = models[k]
    if os.path.exists(os.path.join(OUT, name + ".json")):
        continue
    _count["anchors"] = 0; t0 = time.time()
    acc, ev_estpy, e23, e12, be = f(m)
    flops = _count["anchors"] * ANCHOR_FLOPS + 2 * 720 * q * nodes * 41 + 10 * FPE
    j = dict(name=name, cfg=[nv, ng, q], est=float(acc), err23=float(e23), err12=float(e12), blind_err=float(be),
             anchors=_count["anchors"], flops=float(flops), C_fpe=flops / FPE, C_fpe_estpy=float(ev_estpy), sec=time.time() - t0)
    json.dump(j, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(k, name, "est %.5f e23 %.4f e12 %.4f blind %.3f  C %.0f FPE (est.py %.0f)  %.0fs" % (
        acc, e23, e12, be, j["C_fpe"], ev_estpy, j["sec"]), flush=True)

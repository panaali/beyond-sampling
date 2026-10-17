"""T87 diagnosis on zoo_2nd_argmax(4,4) seed 0: which approximation causes the recipe's error, and where the ops go.
usage: OMP_NUM_THREADS=1 python diag87.py   -> diag87.log"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo

m = zoo(4, 4, 0)
o = Ops(); ex = estT.estT(m, o, mode="exact")
# ops profile of exact mode
prof = {}
for name in ("_adjacent_pairs", "orthant", "clip", "mass"):
    f = getattr(estT, name)
    def wrap(*a, _f=f, _n=name, **k):
        ops = [x for x in a if isinstance(x, Ops)][0]; b = ops.n; r = _f(*a, **k); prof[_n] = prof.get(_n, 0) + ops.n - b; return r
    setattr(estT, name, wrap)
o = Ops(); estT.estT(m, o, mode="exact")
print(f"exact {ex:.5f}, {o.n/1e6:.1f}M ops; inclusive profile (M ops): " + ", ".join(f"{k} {v/1e6:.1f}" for k, v in prof.items()), flush=True)
for name in list(prof):
    pass
import importlib; importlib.reload(estT)
orig_split = estT._split_impact
for label, kw, patch in (("recipe eps=0.1", dict(eps=0.1), None),
                          ("eps=0.01", dict(eps=0.01), None),
                          ("intermediate steps exact, last step margin eps=0.1", dict(eps=0.1), "inter_exact"),
                          ("intermediate eps=0.1, last step exact", dict(eps=0.1), "last_exact")):
    if patch == "inter_exact":
        estT._split_impact = lambda cells, A, w, eps, ops, exact: orig_split(cells, A, w, eps, ops, True)
    elif patch == "last_exact":
        om = estT._margin_cells
        estT._margin_cells = lambda c, A, Wo, truth, n_true, eps, ops, exact, done=(): om(c, A, Wo, truth, n_true, eps, ops, True, done)
    estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0); o = Ops(); t0 = time.time()
    e = estT.estT(m, o, **kw)
    print(f"{label:55s} err {100*(e-ex):+.3f}pp ops {o.n/1e6:.1f}M cells {estT.STATS['cells']} simplices {estT.STATS['simplices']} ({time.time()-t0:.0f}s)", flush=True)
    importlib.reload(estT); orig_split = estT._split_impact

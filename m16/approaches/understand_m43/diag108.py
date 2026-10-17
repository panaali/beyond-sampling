"""T108 diagnosis on zoo (8,4) seed 0: exact enumeration (truth) and the recipe with one step resolved exactly at
a time, to attribute the recipe's error to steps. usage: OMP_NUM_THREADS=1 python diag108.py [seed] -> diag108_<seed>.log"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import estT
from m43 import Ops
from zoo44 import zoo

sd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
m = zoo(8, 4, sd); res = {}
for label, mode, ex in (("exact", "exact", set()), ("recipe108", "recipe108", set()), ("step t=1 exact", "recipe108", {1}),
                        ("step t=2 exact", "recipe108", {2}), ("step t=3 (last) exact", "recipe108", {3})):
    estT.EXACT_STEPS[0] = ex; estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0)
    o = Ops(); t0 = time.time(); a = estT.estT(m, o, mode=mode)
    res[label] = dict(acc=a, ops=o.n, cells=estT.STATS["cells"], sec=time.time() - t0)
    print(f"{label:24s} acc {a:.6f} err vs exact {100*(a-res['exact']['acc']):+.4f}pp ops {o.n/1e6:.0f}M cells {estT.STATS['cells']} ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"diag108_{sd}.json"), "w"), indent=1)

"""Exp 2 (partial): estT has no 5-D cone mass (mass() asserts d==4), so at T=5 we run the recipe108 splitting with
mass() stubbed to 0 and count recipe cells + non-mass ops (lower bound on cost). usage: python cells5.py H T seed"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [U, os.path.join(U, "..", "arc_m43_repro")]
import estT
from m43 import Ops
from zoo44 import zoo
from run_regions import regions
H, T, s = map(int, sys.argv[1:4]); mode = sys.argv[4] if len(sys.argv) > 4 else "recipe108"
estT.mass = lambda c, ops: 0.0
estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0); o = Ops(); t0 = time.time()
estT.estT(zoo(H, T, s), o, mode=mode); m = zoo(H, T, s)
fn = os.path.join(HERE, "cells_nomass.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
r = out[f"H{H}_T{T}_zoo{s}_{mode}"] = dict(cells=estT.STATS["cells"], ops_nomass=o.n, regions_lb=regions(m, T), sec=time.time() - t0)
json.dump(out, open(fn, "w"), indent=1); print(f"H{H} T{T} zoo{s} {mode}: cells {r['cells']} ops(no mass) {o.n:.3g} reg>={r['regions_lb']} {r['sec']:.0f}s")

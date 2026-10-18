"""T205: RECIPE112 (understand_m43/est75.py, unchanged, imported read-only) clean-seed cell counts at (H,3), counted
once per run (evalH.py lines 34-38 add the rescale re-run of model 0 into STATS, doubling E43's clean_cells).
usage: python recount112.py -> recount112.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [U, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "understand_ladder")]
import est75 as E75
from m43 import Ops
from zoo44 import zoo
out = {}
for H in (4, 5, 6, 8, 16):
    ev = json.load(open(os.path.join(U, f"evalH_{H}.json")))["zoo"] if H < 16 else {}
    for s in range(5):
        E75.STATS.update(cut=0, resolved=0, cells=0); o = Ops(); acc = E75.est75(zoo(H, 3, s), o, **E75.RECIPE112)
        out[f"H{H}_T3_zoo{s}"] = dict(cells=E75.STATS["cells"], ops=o.n, acc=acc,
                                      e43_clean_cells=ev.get(str(s), {}).get("clean_cells"), e43_regions=ev.get(str(s), {}).get("regions"))
        print(H, s, out[f"H{H}_T3_zoo{s}"], flush=True)
json.dump(out, open(os.path.join(HERE, "recount112.json"), "w"), indent=1)

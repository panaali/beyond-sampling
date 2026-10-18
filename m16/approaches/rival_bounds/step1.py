"""T289 step 1: enclosure + cost on zoo (4,3) and (4,4) seed 0. -> step1.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "m16", "approaches", "understand_m43")]
import numpy as np
import rb
from zoo44 import zoo, mc_acc
out = {}
for T in (3, 4):
    m = zoo(4, T, 0)
    truth = 0.477013125994119 if T == 4 else mc_acc(m, 10_000_000, np.random.default_rng(2893))
    r = {"truth": truth, "truth_kind": "independent exact reference" if T == 4 else "MC 1e7 (diagnostic)"}
    for b in ("forward", "crown"):
        e = rb.estimate(m, b); e["encloses"] = bool(e["l"] <= truth <= e["u"]); e["width"] = e["u"] - e["l"]
        e["fpe_bounds"] = e["flops_bounds"] / (T * 4 * (4 + 1) * 2 + T * 4 * 2 + 4 * T * 2)       # rough FPE at (4,T)
        r[b] = e; print(T, b, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)
    out[f"H4T{T}"] = r
json.dump(out, open(os.path.join(HERE, "step1.json"), "w"), indent=1)

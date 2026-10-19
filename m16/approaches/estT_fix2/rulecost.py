"""T443 post-hoc sensitivity (NOT pre-registered): how cheap can a converged d=4 'wide' rule be? Old engine
(understand_m43/estT.py) with both rules set to n-node graded GL (p=3); value and op count on one (4,4) model,
against estT_fix exact. usage: OMP_NUM_THREADS=1 python rulecost.py TAG SEED -> rulecost.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(A, "understand_m43"), os.path.join(A, "arc_m43_repro")]
import numpy as np
import estT
from m43 import Ops
tag, s = sys.argv[1], sys.argv[2]
d = np.load(os.path.join(HERE, "inputs", f"m_4_4_{tag}_{s}.npz")); m = (d["Wi"], d["Wh"], d["Wo"])
ref = json.load(open(os.path.join(HERE, "rescore_t111.json")))[f"{tag}_{s}"]
out = {}; saved = dict(estT._RULES)
for n in (12, 16, 24, 32, 48, 128):
    estT._RULES.update(wide=estT._rule(n, 3), thin=estT._rule(max(n, 48), 3))
    for mode in ("exact", "recipe"):
        o = Ops(); p = estT.estT(m, o, mode=mode); out[f"{n}_{mode}"] = dict(p=p, ops=o.n)
    estT._RULES.update(saved)
    print(n, "exact %.7f (fix %.7f, diff %+.1e) ops %.1fM | recipe %.7f ops %.1fM (fix recipe %.7f, %.1fM)" % (
        out[f"{n}_exact"]["p"], ref["exact_fix"], out[f"{n}_exact"]["p"] - ref["exact_fix"], out[f"{n}_exact"]["ops"] / 1e6,
        out[f"{n}_recipe"]["p"], out[f"{n}_recipe"]["ops"] / 1e6, ref["recipe_fix"], ref["ops_fix"] / 1e6), flush=True)
json.dump(out, open(os.path.join(HERE, f"rulecost_{tag}_{s}.json"), "w"), indent=1)

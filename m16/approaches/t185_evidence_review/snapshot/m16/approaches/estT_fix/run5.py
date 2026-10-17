"""T185: estT (fixed, 5-D masses) at (8,5). usage: python run5.py name mode [leafsum] -> run5/<name>_<mode>.json
name zoo<s> or zoo0_noisy<i> (final3/leaderboard _noisy(zoo0, 4, .005, 7)); mode recipe108 | exact."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np, estT
from m43 import Ops
from zoo44 import zoo
from leaderboard import _noisy
name, mode = sys.argv[1], sys.argv[2]; LS = len(sys.argv) > 3
m = zoo(8, 5, int(name[3])) if "noisy" not in name else _noisy(zoo(8, 5, 0), 4, 0.005, 7)[int(name[-1])]
LEAF = [0.0]
if LS:
    ce0 = estT._cell_error
    def ce(c, g, A, Wo, truth, ops):
        LEAF[0] += estT.mass(c, Ops()); return ce0(c, g, A, Wo, truth, ops)
    estT._cell_error = ce
o = Ops(); t0 = time.time(); acc = estT.estT(m, o, mode=mode)
r = dict(acc=acc, ops=o.n, sec=time.time() - t0, **estT.STATS)
if LS: r["leafsum_minus_1"] = LEAF[0] - 1
os.makedirs(os.path.join(HERE, "run5"), exist_ok=True)
json.dump(r, open(os.path.join(HERE, "run5", f"{name}_{mode}.json"), "w"), indent=1)
print(name, mode, {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

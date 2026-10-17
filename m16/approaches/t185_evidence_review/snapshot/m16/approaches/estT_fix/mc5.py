"""1e7 MC truth (rng 2026) at (8,5) for the given names -> mc_H8_T5.json (merged)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
from zoo44 import zoo, mc_acc
from leaderboard import _noisy
fn = os.path.join(HERE, "mc_H8_T5.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
for n in sys.argv[1:]:
    m = zoo(8, 5, int(n[3])) if "noisy" not in n else _noisy(zoo(8, 5, 0), 4, 0.005, 7)[int(n[-1])]
    out[n] = mc_acc(m, 10_000_000, np.random.default_rng(2026), chunk=500_000)
    json.dump(out, open(fn, "w"), indent=1); print(n, out[n])

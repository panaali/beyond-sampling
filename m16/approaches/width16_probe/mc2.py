"""T194: second, independent MC check (rng 7, 2e7 samples) for the given H T names -> mc2.json (merged)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "..")]
import numpy as np
from zoo44 import zoo, mc_acc
from leaderboard import _noisy
H, T = int(sys.argv[1]), int(sys.argv[2]); fn = os.path.join(HERE, "mc2.json")
out = json.load(open(fn)) if os.path.exists(fn) else {}
for n in sys.argv[3:]:
    k = f"H{H}_T{T}_{n}"
    if k not in out:
        m = zoo(H, T, int(n[3])) if "noisy" not in n else _noisy(zoo(H, T, 0), 4, 0.005, 7)[int(n[-1])]
        out[k] = mc_acc(m, 20_000_000, np.random.default_rng(7), chunk=500_000)
        json.dump(out, open(fn, "w"), indent=1); print(k, out[k], flush=True)

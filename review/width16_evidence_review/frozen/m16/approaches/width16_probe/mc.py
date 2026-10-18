"""T194: 1e7 MC truth (rng 2026, chunks of 5e5; diagnostic only) and region lower bounds (distinct gate patterns over
all steps among n N(0,I) inputs, rng 5 = final3/run_regions convention; n = 2e5 and 2e6) for zoo (H,T) names.
usage: python mc.py H T names... -> mc.json (merged, resumable)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "..")]
import numpy as np
from zoo44 import zoo, mc_acc
from leaderboard import _noisy


def regions(m, T, n, seed=5, chunk=200_000):
    Wi, Wh, Wo = m; H = len(Wi); rng = np.random.default_rng(seed); seen = set()
    for i in range(0, n, chunk):
        x = rng.standard_normal((min(chunk, n - i), T)); h = np.zeros((len(x), H)); G = []
        for t in range(T):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); G.append(h > 0)
        seen.update(map(bytes, np.packbits(np.concatenate(G, 1), 1)))
    return len(seen)


H, T = int(sys.argv[1]), int(sys.argv[2]); fn = os.path.join(HERE, "mc.json")
out = json.load(open(fn)) if os.path.exists(fn) else {}
for n in sys.argv[3:]:
    k = f"H{H}_T{T}_{n}"
    if k in out:
        continue
    m = zoo(H, T, int(n[3])) if "noisy" not in n else _noisy(zoo(H, T, 0), 4, 0.005, 7)[int(n[-1])]
    out[k] = dict(mc1e7=mc_acc(m, 10_000_000, np.random.default_rng(2026), chunk=500_000),
                  regions_lb_2e5=regions(m, T, 200_000), regions_lb_2e6=regions(m, T, 2_000_000))
    json.dump(out, open(fn, "w"), indent=1); print(k, out[k], flush=True)

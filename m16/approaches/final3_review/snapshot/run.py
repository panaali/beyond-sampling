"""final3: frozen T108 recipe (estT mode 'recipe108' = RECIPE112 generalised: margin-split with the multi-step
Lipschitz certificate at every step) on zoo (H,T). usage: python run.py H T [seeds=0,1,2,3,4] [noisy=0]
-> res_H{H}_T{T}.json (appends per model). Truth: plain MC 4e6 (seed 99). Regions: distinct gate patterns in 2e5."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo, mc_acc
from leaderboard import fpe_flops, _noisy

H, T = int(sys.argv[1]), int(sys.argv[2])
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2,3,4").split(",")]
NOISY = int(sys.argv[4]) if len(sys.argv) > 4 else 0
fn = os.path.join(HERE, f"res_H{H}_T{T}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}


def regions(m, n=200_000, rng=np.random.default_rng(5)):
    Wi, Wh, Wo = m; x = rng.standard_normal((n, T)); h = np.zeros((n, H)); G = []
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); G.append(h > 0)
    return len(np.unique(np.packbits(np.concatenate(G, 1), 1), axis=0))


models = [(f"zoo{s}", zoo(H, T, s)) for s in seeds]
if NOISY:
    models += [(f"zoo0_noisy{i}", m) for i, m in enumerate(_noisy(zoo(H, T, 0), NOISY, 0.005, 7))]
for name, m in models:
    if name in out:
        continue
    truth = mc_acc(m, 4_000_000, np.random.default_rng(99), chunk=500_000)
    estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0); o = Ops(); t0 = time.time()
    est = estT.estT(m, o, mode="recipe108")
    N = o.n / fpe_flops(H, T); smse = truth * (1 - truth) / N; se2 = truth * (1 - truth) / 4e6
    r = dict(truth=truth, est=est, err_pp=100 * (est - truth), ops=o.n, N_equiv=N,
             ratio=max((est - truth) ** 2 - se2, 0) / smse, ratio_raw=(est - truth) ** 2 / smse,
             cells=estT.STATS["cells"], regions_lb=regions(m), sec=time.time() - t0)
    out[name] = r; json.dump(out, open(fn, "w"), indent=1)
    print(f"{name} err {r['err_pp']:+.4f}pp ops {o.n:.3g} ratio {r['ratio_raw']:.3g} cells {r['cells']} reg>={r['regions_lb']} {r['sec']:.0f}s")

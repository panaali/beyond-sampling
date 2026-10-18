"""T271 D6: active-fraction-matched random controls for T232 (pre-registered in README).
pool: 400 Frobenius-matched i.i.d. draws (seeds 1000..1399, as random_control frob), T = 10 active fraction from 1e5 inputs
(mean of h_T > 0, rng 7, the T232 definition). select: the 5 closest to the target's. count: exact prefix regions k = 2..4.
usage: python d6.py pool | pool_mu | count -> d6.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "width16_probe"), os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
from prefix import target, run
fn = os.path.join(HERE, "d6.json"); D = json.load(open(fn)) if os.path.exists(fn) else {}
Wi0, Wh0, Wo0 = target(); X = np.random.default_rng(7).standard_normal((100_000, 10))


def draw(seed, mu=0.0):
    r = np.random.default_rng(seed)
    f = lambda W, m=0.0: (lambda Z: Z * np.linalg.norm(W) / np.linalg.norm(Z))(r.standard_normal(W.shape) + m)
    return f(Wi0), f(Wh0, mu), f(Wo0)


def active(m):
    Wi, Wh, _ = m; h = np.zeros((len(X), len(Wi)))
    for t in range(10):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return float((h > 0).mean())


MU = 0.25        # fallback per pre-registration: grid 0/.1/.2/.3/.4/.6 gave median activity .51/.61/.69/.82/.88/1.0 (30 draws each)
if sys.argv[1] == "pool_mu":
    D["mu"] = MU; D["pool_mu"] = {s: active(draw(s, MU)) for s in range(3000, 3100)}
    a = np.array(list(D["pool_mu"].values())); print("mu pool quantiles", np.quantile(a, [0, .1, .5, .9, 1]).round(3))
    close = sorted(D["pool_mu"], key=lambda s: abs(D["pool_mu"][s] - D["target_active"]))[:5]
    D["selected_mu"] = [int(s) for s in close]; print("selected_mu", [(s, round(D["pool_mu"][s], 3)) for s in close])
elif sys.argv[1] == "pool":
    D["target_active"] = active((Wi0, Wh0, Wo0))
    D["pool"] = {s: active(draw(s)) for s in range(1000, 1400)}
    a = np.array(list(D["pool"].values())); print("target", D["target_active"], "pool quantiles", np.quantile(a, [0, .1, .5, .9, 1]).round(3))
    close = sorted(D["pool"], key=lambda s: abs(D["pool"][s] - D["target_active"]))[:5]
    D["selected"] = [int(s) for s in close]; print("selected", [(s, round(D["pool"][s], 3)) for s in close])
else:
    out = {}
    for s in [f"mu{x}" for x in D["selected_mu"]] + [str(x) for x in D["selected"][:1]]:
        m = draw(int(s[2:]), MU) if s.startswith("mu") else draw(int(s))
        out[str(s)] = [run(m, k)[0]["regions"] for k in (2, 3, 4)]; print(s, out[str(s)], flush=True)
    D["regions"] = out
json.dump(D, open(fn, "w"), indent=1)

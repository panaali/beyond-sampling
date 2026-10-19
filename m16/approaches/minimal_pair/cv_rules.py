"""T380 post-hoc secondary (declared after the leaderboard T=6 rows): 32 alternative fixed rules of the frozen cv
recipe on zoo(16,6) seeds 0-4, vs own 48M truths; compare with mse_T6.json (T10 recipe, same protocol)."""
import importlib.util, json, sys
import numpy as np
import core as C
spec = importlib.util.spec_from_file_location("cvE", "cv/estimator.py"); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
T = int(sys.argv[1]); R = int(sys.argv[2])
tr = json.load(open("truths.json")); base = json.load(open(f"mse_T{T}.json"))
out = {}
for s in range(5):
    m = C.zoo(16, T, s); p, se = tr[f"16:{T}:{s}"]["acc"], tr[f"16:{T}:{s}"]["se"]
    for b in (16000, 64000):
        r = [E.est(m, b, rule_seed=7000 + 13 * k) for k in range(R)]
        out[f"{s}|cv|{b}"] = dict(est=[e for e, _ in r], cost=[c for _, c in r])
        g = {}
        for v, d in (("cv", out[f"{s}|cv|{b}"]), ("frozen", base[f"{s}|frozen|{b}"])):
            e = np.array(d["est"]); c = np.array(d["cost"])
            g[v] = np.mean(p * (1 - p) / c) / max(np.mean((e - p) ** 2) - se ** 2, 1e-30)
        print(f"T{T} s{s} {b}: gain frozen {g['frozen']:.1f}x  cv {g['cv']:.1f}x  ratio cv/frozen {g['cv']/g['frozen']:.2f}", flush=True)
json.dump(out, open(f"cv_rules_T{T}.json", "w"))

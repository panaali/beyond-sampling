"""Score leaderboard-style estimators on DEV3 (100 open M_16,10 retrains, 1e8-sample truths): one deterministic run per
model per budget; MSE ratio = mean((est-p)^2 - se^2) / mean(p(1-p)/cost), cluster bootstrap over models."""
import importlib.util, json, sys, time
import numpy as np
names = sys.argv[1].split(","); budgets = [int(b) for b in sys.argv[2].split(",")]
W = np.load("../../../data/dev3/weights.npz"); tr = json.load(open("../../../data/dev3/truths.json"))["truths"]
mods = {str(n): (W["Wi"][i], W["Wh"][i], W["Wo"][i]) for i, n in enumerate(W["names"])}
out = {}
for name in names:
    path = f"{name}/estimator.py" if name != "minimal_pair" else "estimator.py"
    spec = importlib.util.spec_from_file_location(name.replace("/", "_"), path); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    for b in budgets:
        rows = []
        for mn, m in mods.items():
            t0 = time.process_time(); e, c = E.est(m, b, None)
            rows.append(dict(model=mn, est=e, cost=c, cpu=time.process_time() - t0))
        out[f"{name}|{b}"] = rows
        p = np.array([tr[r["model"]]["acc"] for r in rows]); se = np.array([tr[r["model"]]["se"] for r in rows])
        e = np.array([r["est"] for r in rows]); c = np.array([r["cost"] for r in rows])
        se2 = (e - p) ** 2 - se ** 2; v = p * (1 - p) / c
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            ii = rng.integers(0, len(p), len(p)); bs.append(se2[ii].mean() / v[ii].mean())
        print(f"{name} {b}: cost {c.mean():.0f}  MSE ratio {se2.mean()/v.mean():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]"
              f"  bias {100*(e-p).mean():+.3f} pp  RMSE {100*np.sqrt(max(se2.mean(),0)):.3f} pp", flush=True)
        json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else "dev3_scores.json", "w"))

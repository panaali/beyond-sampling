"""T277 fit (pre-registered rule in README): alpha_rmse(T) per seed and pooled, the s/(T-1) fit, Spearman rho, both bars.
-> fit.json, printed table"""
import json, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "rows.jsonl"))]
TR = json.load(open(os.path.join(HERE, "truths.json")))
G = defaultdict(lambda: defaultdict(list))
for r in rows:
    G[(r["T"], r["seed"])][r["budget"]].append(r)


def slope(C, y):
    lc, ly = np.log(C), np.log(y); res = stats.linregress(lc, ly); t = stats.t.ppf(0.975, len(lc) - 2)
    return dict(alpha=-res.slope, ci=[-res.slope - t * res.stderr, -res.slope + t * res.stderr], n=len(lc))


out = {"per_seed": {}, "pooled": {}}
for T in sorted({k[0] for k in G}):
    pool = defaultdict(list); costs = defaultdict(list)
    for s in sorted({k[1] for k in G if k[0] == T}):
        byb = G[(T, s)]; bs = sorted(byb); C, M, R = [], [], []
        p = TR[f"T{T}_s{s}"]["acc"]
        for b in bs:
            rr = byb[b]
            if len(rr) < 8:
                continue
            mse = np.mean([x["err"] ** 2 - x["truth_se"] ** 2 for x in rr]); c = np.mean([x["cost"] for x in rr])
            pool[b].append(mse); costs[b].append(c)
            if mse > 0:
                C.append(c); M.append(mse); R.append(mse * c / (p * (1 - p)))
        if len(C) >= 3:
            out["per_seed"][f"T{T}_s{s}"] = dict(fit=slope(C, np.sqrt(M)), ratio=R, dropped=len(bs) - len(C))
    bs = [b for b in sorted(pool) if len(pool[b]) == len(costs[b]) and np.mean(pool[b]) > 0]
    if len(bs) >= 3:
        C = [np.mean(costs[b]) for b in bs]; M = [np.mean(pool[b]) for b in bs]
        out["pooled"][T] = dict(fit=slope(C, np.sqrt(M)), cost=C, mse=M)
for T, o in out["pooled"].items():
    f = o["fit"]; ps = [v["fit"]["alpha"] for k, v in out["per_seed"].items() if k.startswith(f"T{T}_")]
    rat = [x for k, v in out["per_seed"].items() if k.startswith(f"T{T}_") for x in v["ratio"]]
    print(f"T={T}: pooled alpha_rmse {f['alpha']:.2f} [{f['ci'][0]:.2f},{f['ci'][1]:.2f}]  per-seed {[round(a, 2) for a in ps]}  "
          f"MSE ratio range {min(rat):.3g}-{max(rat):.3g}" if rat else "")
Ts = np.array(sorted(out["pooled"])); A = np.array([out["pooled"][T]["fit"]["alpha"] for T in Ts])
if len(Ts) >= 4:
    rho = stats.spearmanr(Ts, A).correlation
    x = 1.0 / (Ts - 1); y = A - 0.5; sres = np.sum(x * y) / np.sum(x * x)          # alpha = 0.5 + s/(T-1), no intercept
    resid = y - sres * x; se = np.sqrt(np.sum(resid ** 2) / (len(x) - 1) / np.sum(x * x)); t = stats.t.ppf(0.975, len(x) - 1)
    out["summary"] = dict(rho=float(rho), s=float(sres), s_ci=[float(sres - t * se), float(sres + t * se)], alphas=dict(zip(map(int, Ts), map(float, A))))
    print("Spearman rho %.2f; s = %.2f [%.2f, %.2f]" % (rho, sres, sres - t * se, sres + t * se))
json.dump(out, open(os.path.join(HERE, "fit.json"), "w"), indent=1, default=float)

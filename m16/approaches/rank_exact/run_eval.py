"""ARC-protocol evaluation of the rank-ordering estimator vs sampling (harness: m16/arc_eval.py)."""
import sys, json; sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '/home/user/clean-neurons')
import numpy as np
from arc_eval import noisy_models, true_acc
from rank_est import est_rank
out = {}
for sigma in (0.0025, 0.005):
    ms = noisy_models(6, sigma)
    rows = []
    for i, m in enumerate(ms):
        tr = true_acc(m, n=4_000_000)
        r = {"truth": tr}
        for Q in (16, 64, 1024):
            e, c = est_rank(m, Q=Q); r[f"Q{Q}"] = e; r[f"cost{Q}"] = c
        rows.append(r); print(sigma, i, {k: round(v, 5) for k, v in r.items()}, flush=True)
    res = {}
    for Q in (16, 64, 1024):
        err = np.array([r[f"Q{Q}"] - r["truth"] for r in rows]); cost = np.mean([r[f"cost{Q}"] for r in rows])
        samp = np.sqrt(np.mean([r["truth"] * (1 - r["truth"]) for r in rows]) / cost)
        res[Q] = dict(rmse_pp=100 * float(np.sqrt((err ** 2).mean())), mean_err_pp=100 * float(err.mean()),
                      evals=float(cost), sampling_rmse_pp_same_compute=100 * float(samp))
    print(sigma, json.dumps(res, indent=1), flush=True)
    out[sigma] = {"rows": rows, "summary": res}
json.dump(out, open('eval_results.json', 'w'), indent=1)

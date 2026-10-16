"""ARC-protocol evaluation of est_boundary vs plain sampling (m16/arc_eval.py harness)."""
import sys, json, os, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from arc_eval import noisy_models, true_acc, est_sampling
from estimator import est_boundary
import time as _t
sigma = float(sys.argv[1]); budgets = [4000, 16000, 64000]; reps = 4
tf = f"truth_sigma{sigma}.json"
models = noisy_models(6, sigma)
LB = "/home/user/clean-neurons/results/leaderboard_truths.json"
if os.path.exists(tf):
    truth = json.load(open(tf))
elif os.path.exists(LB):     # DRI's cached truths: same noisy_models(6, sigma, seed=0), 4M samples
    d = json.load(open(LB)); truth = [d[f"s{sigma}_{k}"]["acc"] for k in range(6)]; json.dump(truth, open(tf, "w"))
else:
    truth = [true_acc(m, n=4_000_000) for m in models]; json.dump(truth, open(tf, "w"))
print("truth", truth, flush=True)
res = {"sigma": sigma, "truth": truth, "rows": []}
for b in budgets:
    for name, est in [("boundary", est_boundary), ("sampling", est_sampling)]:
        se, used, errs = [], [], []
        for mi, m in enumerate(models):
            for r in range(reps):
                e, u = est(m, b, np.random.default_rng(1000 * mi + r))
                se.append((e - truth[mi]) ** 2); used.append(u); errs.append(e - truth[mi])
        an = np.sqrt(np.mean([p * (1 - p) for p in truth]) / np.mean(used))
        row = dict(est=name, budget=b, evals=float(np.mean(used)), rmse=float(np.sqrt(np.mean(se))),
                   mean_err=float(np.mean(errs)), sampling_rmse_analytic_at_evals=float(an))
        res["rows"].append(row); print(row, flush=True)
json.dump(res, open(f"arc_eval_sigma{sigma}.json", "w"), indent=1)

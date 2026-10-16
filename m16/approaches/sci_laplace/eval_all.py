"""Evaluate LTQ on the leaderboard's 18 models (truths: results/leaderboard_truths.json, 4M samples).
S point-set randomisations per model give the RMSE of the rule class; seed 0 is the deterministic entry.
Sampling RMSE is analytic, sqrt(p(1-p)/cost), at the same cost."""
import sys, json, os
import numpy as np
os.chdir("/home/user/clean-neurons")
sys.path.insert(0, "m16")
from leaderboard import models
sys.path.insert(0, "m16/approaches/sci_laplace")
from estimator import est_seeded
truth = json.load(open("results/leaderboard_truths.json"))
B = int(sys.argv[1]); S = int(sys.argv[2])
out = {}
for name, m in models().items():
    p = truth[name]["acc"]
    r = [est_seeded(m, B, s) for s in range(S)]
    e = np.array([x[0] for x in r]) - p; c = np.mean([x[1] for x in r])
    samp = np.sqrt(p * (1 - p) / c)
    out[name] = dict(err=e.tolist(), cost=c, samp_rmse=samp)
    print("%-10s acc %.4f  seed0 err %+.3fpp  mean err %+.3fpp  RMSE %.3fpp  sampling %.3fpp  MSE ratio %.2f  cost %.0f"
          % (name, p, 100 * e[0], 100 * e.mean(), 100 * np.sqrt((e ** 2).mean()), 100 * samp,
             (e ** 2).mean() / samp ** 2, c), flush=True)
json.dump(out, open(f"m16/approaches/sci_laplace/eval_{B}.json", "w"), indent=1)

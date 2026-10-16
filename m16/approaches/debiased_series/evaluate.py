"""Compute-adjusted MSE vs plain sampling (analytic: a(1-a)/B, exact for a Bernoulli mean) and vs stratified.py.
usage: python evaluate.py {0|0.0025|0.005} REPS"""
import sys, os, json, time, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_eval import noisy_models
from common import Wi, Wh, Wo
from ds import est_ds, est_strat, HERE

which, reps = sys.argv[1], int(sys.argv[2])
truth = [0.95317945] if which == "0" else json.load(open(os.path.join(HERE, "truth.json")))[which]
models = [(Wi, Wh, Wo)] if which == "0" else noisy_models(6, float(which))
methods = {"strat": est_strat, "ds3": lambda m, b, r: est_ds(m, b, r, 3), "ds5": lambda m, b, r: est_ds(m, b, r, 5)}
out = {"set": which, "truth": truth, "reps": reps, "rows": []}
for B in (16_000, 64_000, 256_000):
    for name, est in methods.items():
        errs, used = [], []
        t0 = time.time()
        for mi, m in enumerate(models):
            for r in range(reps):
                e, u = est(m, B, np.random.default_rng(10_000 * mi + 97 * r + B))
                errs.append(e - truth[mi]); used.append(u)
        errs = np.array(errs); cost = float(np.mean(used))
        plain = float(np.mean([t * (1 - t) for t in truth])) / cost     # plain sampling MSE at the same cost
        mse = float(np.mean(errs ** 2))
        row = {"B": B, "method": name, "cost": cost, "mse": mse, "plain_mse_same_cost": plain,
               "gain_vs_plain": plain / mse, "mean_err": float(errs.mean()),
               "se_mean_err": float(errs.std(ddof=1) / np.sqrt(len(errs))), "n": len(errs)}
        out["rows"].append(row)
        print(f"B={B:>6} {name:6s} cost={cost:8.0f} rmse={np.sqrt(mse)*100:.4f}pp gain_vs_plain={plain/mse:5.2f} "
              f"bias={errs.mean()*100:+.4f}+-{row['se_mean_err']*100:.4f}pp ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(HERE, f"results_{which}.json"), "w"), indent=1)

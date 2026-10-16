"""ARC-protocol evaluation. usage: python run_eval.py <sigma|0> <estimator> <budgets comma> <reps>
sigma 0 = original model (truth 0.953179, n=2e7). Prints MSE, mean evals, and the compute-adjusted gain
G = [mean_m a_m(1-a_m) / evals] / MSE  (plain sampling's exact expected MSE at the same evals over the same copies)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import estimators as E
from arc_eval import noisy_models
from common import Wi, Wh, Wo
sig, name = sys.argv[1], sys.argv[2]
budgets = [int(b) for b in sys.argv[3].split(",")]; reps = int(sys.argv[4])
if float(sig) == 0:
    models, truth = [(Wi, Wh, Wo)], [0.95317945]
else:
    models = noisy_models(6, float(sig)); truth = json.load(open(os.path.join(HERE, "truth.json")))[sig]
est = getattr(E, "est_" + name)
out = []
for b in budgets:
    t0 = time.time(); err, used = [], []
    for mi, m in enumerate(models):
        for r in range(reps):
            e, u = est(m, b, np.random.default_rng(10_000 * mi + 77 * r + b))
            err.append(e - truth[mi]); used.append(u)
    err = np.array(err); ev = np.mean(used); mse = np.mean(err ** 2)
    ref = np.mean([a * (1 - a) for a in truth]) / ev
    se_g = (ref / mse) * np.sqrt(2 / len(err))       # rough: relative s.e. of an MSE from n draws ~ sqrt(2/n)
    row = dict(sigma=sig, est=name, budget=b, evals=float(ev), mse=float(mse), bias=float(err.mean()),
               bias_se=float(err.std() / np.sqrt(len(err))), gain=float(ref / mse), gain_se=float(se_g), n=len(err))
    print(json.dumps(row), "%.0fs" % (time.time() - t0), flush=True); out.append(row)
with open(os.path.join(HERE, "results.jsonl"), "a") as f:
    for row in out: f.write(json.dumps(row) + "\n")

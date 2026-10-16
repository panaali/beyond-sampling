"""ARC protocol: RMSE of the propagation estimator on noisy copies vs sampling at equal compute."""
import sys, json, time
sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '/home/user/clean-neurons')
import numpy as np
from arc_eval import noisy_models, true_acc
from common import Wi, Wh, Wo
from gsf2 import gsf2
F1 = 10 * (2 * 16 * 16 + 3 * 16) + 2 * 10 * 16      # FLOPs of one 10-step forward pass (5,920)
settings = [(1, 1e-3), (1, 1e-4)]
out = {}
for sigma in [0.0025, 0.005]:
    models = noisy_models(6, sigma)
    truth = [true_acc(m, n=4_000_000) for m in models]
    for K, eps in settings:
        est, passes = [], []
        for m in models:
            r = gsf2(m, K=K, eps=eps); est.append(r['acc']); passes.append(r['flops'] / F1)
        e = np.array(est) - np.array(truth)
        n = np.mean(passes); p = np.array(truth)
        rmse_s = np.sqrt(np.mean(p * (1 - p)) / n)
        row = dict(sigma=sigma, K=K, eps=eps, passes=n, rmse=float(np.sqrt(np.mean(e ** 2))), rmse_sampling=float(rmse_s),
                   truth=truth, est=est)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k not in ('truth', 'est')}),
              'truth', np.round(truth, 4), 'est', np.round(est, 4), flush=True)
        out[f'{sigma}_{K}_{eps}'] = row
json.dump(out, open('results_arc.json', 'w'), indent=1)

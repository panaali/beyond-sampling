"""ARC protocol: deterministic tube estimate for the original + noisy_models(6, 0.0025/0.005), budgets 1k-64k."""
import sys, json, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from arc_eval import noisy_models, Wi0, Wh0, Wo0
from estimator import est_tube
HERE = os.path.dirname(os.path.abspath(__file__))
models = {"orig": (Wi0, Wh0, Wo0)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s)):
        models[f"{s}_{k}"] = m
out = {}
for B in (1000, 4000, 16000, 64000):
    for name, m in models.items():
        t = time.time(); a, u, p = est_tube(m, B, parts=True)
        out[f"{name}|{B}"] = dict(est=float(a), used=float(u), **{k: float(v) for k, v in p.items()})
        print(name, B, round(a, 5), round(u), round(time.time() - t, 1), flush=True)
    json.dump(out, open(os.path.join(HERE, "eval_estimates.json"), "w"), indent=1)

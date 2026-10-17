"""Held-out check: noise seeds 4 and 5 (2 x 256 fresh noisy copies, never looked at while designing est_mech).
Truth by exact enumeration (arc_m43_repro/exact.py). usage: OMP_NUM_THREADS=1 python heldout.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import numpy as np
import est_mech as EM
from m43 import Ops, noisy_models, OPS_PER_SAMPLE
from exact import exact_acc
from run_mech import TIERS

models = noisy_models(256, 0.05, seed=4) + noisy_models(256, 0.05, seed=5)
truth = np.array([exact_acc(m) for m in models]); pq = float(np.mean(truth * (1 - truth)))
out = {"mean_truth": float(truth.mean()), "pq": pq, "tiers": {}}
for name in ("A anchor pair, anchor strip", "B default (A2 + resolved crossover)", "C resolved everywhere"):
    kw = dict(TIERS[name]); EM.COMPLETE = kw.pop("complete")
    est = []; ops = []
    for m in models:
        o = Ops(); est.append(EM.est_mech(m, o, **kw)); ops.append(o.n)
    e = np.array(est) - truth; om = float(np.mean(ops)); mse_s = pq / (om / OPS_PER_SAMPLE)
    out["tiers"][name] = {"rms": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()), "max_abs": float(np.abs(e).max()),
                          "ops": om, "mse_ratio": float(np.mean(e ** 2) / mse_s)}
    print(f"{name:38s} rms {np.sqrt(np.mean(e**2)):.4f} bias {e.mean():+.4f} max {np.abs(e).max():.4f} ops {om:.0f} "
          f"sampling rmse {np.sqrt(mse_s):.4f} MSE ratio {np.mean(e**2)/mse_s:.4f}")
print("mean truth", truth.mean())
json.dump(out, open(os.path.join(HERE, "heldout.json"), "w"), indent=1)

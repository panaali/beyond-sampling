"""Low-noise comparison: predicted MSE = sum_h P_h^2 sd_h^2 / n_h from per-stratum sds measured on 300k
inputs per model (repeat-based MSE with few reps is too noisy to rank methods that differ by <1.5x).
Same fixed allocations as the estimators. Plain sampling MSE = a(1-a)/B."""
import sys, os, json, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from stratified import stratum, N_STRATA
from surrogate import thresholds, g
from ds import P, DESIGN, residual_alloc, HI0, W, G_COST, HERE
x = np.random.default_rng(4242).standard_normal((300_000, T)); s = stratum(x)
models = [("orig", (Wi, Wh, Wo))] + [(f"s0.0025_{k}", m) for k, m in enumerate(noisy_models(6, .0025))] + \
         [(f"s0.005_{k}", m) for k, m in enumerate(noisy_models(6, .005))]
Bs = (16_000, 64_000, 256_000); out = {}
def mse(r, alloc, nsamp):
    sd = np.array([r[s == h].std() for h in range(N_STRATA)]); n = np.maximum(2, alloc * nsamp)
    return float((P ** 2 * sd ** 2 / n).sum())
for name, m in models:
    f = (forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).astype(float); a = f.mean(); row = {}
    for B in Bs:
        plain = a * (1 - a) / B
        row[f"strat_{B}"] = plain / mse(f, np.array(DESIGN["alloc"]), B)
    for it in (3, 5):
        t23, t12, C = thresholds(m, it, hi0=HI0); r = f - g(x, t23, t12, 0.0, W)
        for B in Bs:
            row[f"ds{it}_{B}"] = a * (1 - a) / B / mse(r, residual_alloc(it), (B - C) / (1 + G_COST))
        row[f"ds{it}_asymptotic"] = a * (1 - a) / mse(r, residual_alloc(it), 1.0) / (1 + G_COST)
    row["strat_asymptotic"] = a * (1 - a) / mse(f, np.array(DESIGN["alloc"]), 1.0)
    out[name] = row
    print(name, " ".join(f"{k}={v:.2f}" for k, v in row.items()), flush=True)
json.dump(out, open(os.path.join(HERE, "variance_gain.json"), "w"), indent=1)

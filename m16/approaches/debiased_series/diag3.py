"""Variance (stratified, per unit sample) of f - g_K for surrogates of increasing depth K (probe cost C_K).
Answers: does a deeper level of the probe hierarchy buy enough residual-variance reduction to pay for itself?"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from stratified import stratum, N_STRATA
from surrogate import thresholds, g
rng = np.random.default_rng(9); n = 200_000
x = rng.standard_normal((n, T)); s = stratum(x); P = np.bincount(s, minlength=N_STRATA) / n
t0 = time.time(); [forward((Wi, Wh, Wo), x) for _ in range(3)]; tf = (time.time() - t0) / 3
t23, t12, _ = thresholds((Wi, Wh, Wo), iters=2)
t0 = time.time(); [g(x, t23, t12, 0.22, 0.6) for _ in range(3)]; tg = (time.time() - t0) / 3
print(f"time per 200k: forward {tf:.3f}s, surrogate g {tg:.3f}s (ratio {tg/tf:.3f})", flush=True)

def nvar(r):   # Neyman-optimal stratified variance per sample: (sum P_h sd_h)^2
    return sum(P[h] * r[s == h].std() for h in range(N_STRATA)) ** 2

models = [('orig', (Wi, Wh, Wo))] + [(f'n025_{i}', m) for i, m in enumerate(noisy_models(6, .0025)[:2])] + \
         [(f'n05_{i}', m) for i, m in enumerate(noisy_models(6, .005)[:3])]
for name, m in models:
    f = (forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).astype(float)
    line = [f"{name} acc {f.mean():.4f} plain {f.var():.4f} neyman-strat {nvar(f):.4f}"]
    for iters in (3, 5, 8):
        t23, t12, ev = thresholds(m, iters=iters)
        best = min((nvar(f - g(x, t23, t12, 0.22, w)), w) for w in (0.3, 0.6, 1.0))
        line.append(f"K(iters={iters},C={ev}): {best[0]:.4f} (w={best[1]})")
    print(" | ".join(line), flush=True)

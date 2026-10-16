"""How much within-stratum variance does the probe surrogate g remove? (control-variate potential)"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from stratified import stratum, N_STRATA
from surrogate import thresholds, g
rng = np.random.default_rng(8); n = 200_000
x = rng.standard_normal((n, T)); s = stratum(x); P = np.bincount(s, minlength=N_STRATA) / n
for name, m in [('orig', (Wi, Wh, Wo)), ('n05_0', noisy_models(6, .005)[0]), ('n05_3', noisy_models(6, .005)[3])]:
    t0 = time.time(); t23, t12, ev = thresholds(m)
    f = (forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).astype(float)
    print(name, "acc", f.mean().round(4), "probe evals", ev, "t23 med", np.median(t23).round(3),
          "t12 med", np.median(t12).round(3), f"{time.time()-t0:.0f}s", flush=True)
    for w in (0, 0.3, 0.6, 1.0, 2.0):
        gg = g(x, t23, t12, 0.22, w)
        vf = sum(P[h] * f[s == h].var() for h in range(N_STRATA))
        vd = sum(P[h] * (f - gg)[s == h].var() for h in range(N_STRATA))
        vb = 0
        for h in range(N_STRATA):
            k = s == h; c = np.cov(f[k], gg[k]); vb += P[h] * (c[0, 0] - c[0, 1] ** 2 / max(c[1, 1], 1e-12))
        print(f"  w={w}: plain {f.var():.4f} | strat {vf:.4f} | strat+CV(b=1) {vd:.4f} | strat+CV(opt b) {vb:.4f}"
              f" | plain CV(b=1) {(f-gg).var():.4f} | E[f-g] {np.mean(f-gg):+.4f}", flush=True)

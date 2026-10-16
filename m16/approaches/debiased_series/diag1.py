"""Diagnostics: within-stratum variance of f, of threshold-surrogate residual, and of swap-antithetic couplings."""
import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from stratified import stratum, N_STRATA
rng = np.random.default_rng(7); n = 300_000
x = rng.standard_normal((n, T)); o = np.argsort(-x, 1); s = stratum(x)
P = np.bincount(s, minlength=N_STRATA) / n
def swap(x, i, j):
    y = x.copy(); r = np.arange(len(x)); y[r, i], y[r, j] = x[r, j], x[r, i]; return y
def F(m, y): return (forward(m, y).argmax(1) == np.argsort(y, 1)[:, -2]).astype(float)
models = [('orig', (Wi, Wh, Wo)), ('n05_0', noisy_models(6, .005)[0]), ('n05_3', noisy_models(6, .005)[3])]
for name, m in models:
    f = F(m, x)
    f23 = F(m, swap(x, o[:, 1], o[:, 2])); f12 = F(m, swap(x, o[:, 0], o[:, 1]))
    # all 6 perms of top-3 values
    import itertools
    fs = []
    for perm in itertools.permutations(range(3)):
        y = x.copy(); r = np.arange(n)
        for k in range(3): y[r, o[:, k]] = x[r, o[:, perm[k]]]
        fs.append(F(m, y))
    f6 = np.mean(fs, 0)
    print(name, "acc", f.mean().round(4))
    tot = {k: 0 for k in ["f", "a23", "a12", "a6"]}
    for h in range(N_STRATA):
        k = s == h
        v = f[k].var(); va23 = ((f + f23) / 2)[k].var(); va12 = ((f + f12) / 2)[k].var(); v6 = f6[k].var()
        tot["f"] += P[h] * v; tot["a23"] += P[h] * va23 * 2; tot["a12"] += P[h] * va12 * 2; tot["a6"] += P[h] * v6 * 6
        print(f"  h{h} P={P[h]:.4f} err={1-f[k].mean():.3f} Var f={v:.4f}  cost-adj var: anti23={2*va23:.4f} anti12={2*va12:.4f} sym6={6*v6:.4f}")
    print("  strat-weighted cost-adj var (prop. alloc):", {k: round(v, 5) for k, v in tot.items()}, "plain", round(f.var(), 5))

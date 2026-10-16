"""Debiased probe series: two-level (MLMC-degenerate) estimator of accuracy, exactly unbiased for any weights.

    acc = E[g] + E[f - g]

  level 0  E[g]: g is the probe surrogate (surrogate.py). Its thresholds tau come from designed probes on the
           given model (counted forward evals). Given tau, E[g] is a model-free Gaussian integral; because the
           (p1,p2,p3) triple is independent of the order-statistic values and tau lies on the bisection grid,
           E[g] = P(blind)*c + mean_k H[tau23_k, tau12_k] with H tabulated once, model-free (build_table()).
  level 1  E[f - g]: stratified sampling (stratified.py strata) of the residual; f costs 1 eval, g is a lookup
           (measured 0.037 forward-equivalents, counted).

Why not a randomised (Rhee-Glynn) or multi-level version: the exact level f is samplable at unit cost and
all intermediate surrogates are lookups, so any random truncation level N gives an unbiased estimator
E[g_N] + mean(f - g_N); randomising N only mixes the per-depth (cost C_K, residual variance V_K) pairs, and
the optimum of a mixture is at a single depth. See README.
"""
import sys, os, json, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_eval import forward, T
from stratified import stratum, N_STRATA, draw
from surrogate import thresholds, g, feats

HERE = os.path.dirname(os.path.abspath(__file__))
W, HI0, G_COST = 0.6, 0.6, 0.037
DESIGN = json.load(open("/home/user/clean-neurons/m16/stratified_design.json"))
P = np.array(DESIGN["P"])
P_BLIND = 11 / 1024


def grid(iters):
    return (np.arange(2 ** iters) + 0.5) * HI0 / 2 ** iters


def _sig(z): return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def build_table(iters, n=16_000_000, chunk=1_000_000, seed=2024):
    """Model-free H[i,j] = E[ sig((g23-a_i)/(W a_i+1e-3)) * sig((g12-b_j)/(W b_j+1e-3)) ; not blind ]."""
    path = os.path.join(HERE, f"table_it{iters}.npy")
    if os.path.exists(path): return np.load(path)
    a = grid(iters); rng = np.random.default_rng(seed); H = np.zeros((len(a), len(a)))
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T)); _, g12, g23, blind = feats(x); nb = ~blind
        S23 = _sig((g23[nb, None] - a[None]) / (W * a[None] + 1e-3))
        S12 = _sig((g12[nb, None] - a[None]) / (W * a[None] + 1e-3))
        H += S23.T @ S12
    H /= n; np.save(path, H); return H


def residual_alloc(iters, n=300_000, seed=31):
    """Neyman allocation for the residual f-g, from a design-time pilot on the ORIGINAL model only."""
    path = os.path.join(HERE, f"alloc_it{iters}.json")
    if os.path.exists(path): return np.array(json.load(open(path)))
    from common import Wi, Wh, Wo
    m = (Wi, Wh, Wo); t23, t12, _ = thresholds(m, iters, hi0=HI0)
    x = np.random.default_rng(seed).standard_normal((n, T)); s = stratum(x)
    r = (forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]) - g(x, t23, t12, 0.0, W)
    sd = np.array([r[s == h].std() for h in range(N_STRATA)])
    al = P * sd / (P * sd).sum(); json.dump(al.tolist(), open(path, "w")); return al


def est_ds(model, budget, rng, iters=5):
    H = build_table(iters); alloc = residual_alloc(iters); a = grid(iters)
    t23, t12, C = thresholds(model, iters, hi0=HI0)
    i23 = np.rint(t23 / (HI0 / 2 ** iters) - 0.5).astype(int); i12 = np.rint(t12 / (HI0 / 2 ** iters) - 0.5).astype(int)
    assert np.allclose(a[i23], t23) and np.allclose(a[i12], t12)
    Eg = H[i23, i12].mean()                      # + P_BLIND * c with c = 0 (blind constant cancels)
    nsamp = max(0, budget - C) / (1 + G_COST)
    n = np.maximum(2, np.round(alloc * nsamp).astype(int))
    est = Eg
    for h in range(N_STRATA):
        x = draw(h, n[h], rng)
        r = (forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]) - g(x, t23, t12, 0.0, W)
        est += P[h] * r.mean()
    return float(est), C + n.sum() * (1 + G_COST)


def est_strat(model, budget, rng):
    from stratified import estimate
    return estimate(lambda x: forward(model, x), budget, rng, P, np.array(DESIGN["alloc"]))


if __name__ == "__main__":
    for it in (3, 5):
        build_table(it); print(it, residual_alloc(it))

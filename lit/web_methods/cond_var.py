"""Block-preintegration budget test. For a split after step t (prefix x_1..x_t, suffix x_{t+1}..x_10),
estimate Var_prefix(g) with g(prefix)=P(correct | prefix), as a fraction of p(1-p).
If the suffix integral could be done exactly at cost C_in forward-pass equivalents, outer plain MC beats
plain sampling iff C_in < p(1-p)/Var(g) (outer QMC on a smooth g could relax this).
Unbiased: Var(g) = var(row means) - mean(row var)/(m-1)... computed with the standard one-way ANOVA formula.
Usage: OMP_NUM_THREADS=1 python lit/web_methods/cond_var.py [n_prefix] [m_suffix] [seed]"""
import sys, numpy as np
sys.path.insert(0, 'm16')
from common import Wi, Wh, Wo, T
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
m = int(sys.argv[2]) if len(sys.argv) > 2 else 400
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
for t in range(1, 10):
    P = rng.standard_normal((n, t))
    h = np.zeros((n, len(Wi)))
    for k in range(t):
        h = np.maximum(h @ Wh.T + P[:, k:k+1] * Wi, 0)
    S = rng.standard_normal((n, m, T - t))           # independent suffixes per prefix
    hh = np.repeat(h[:, None], m, 1)
    for k in range(T - t):
        hh = np.maximum(hh @ Wh.T + S[:, :, k:k+1] * Wi, 0)
    x = np.concatenate([np.repeat(P[:, None], m, 1), S], 2)
    F = ((hh @ Wo.T).argmax(2) == np.argsort(x, 2)[:, :, -2]).astype(float)
    g = F.mean(1); p = F.mean()
    within = (F.var(1, ddof=1)).mean()
    vg = g.var(ddof=1) - within / m
    print(f"prefix t={t} suffix k={T-t}: p={p:.4f}  Var(g)/p(1-p)={vg/(p*(1-p)):.3f}  "
          f"=> exact-suffix cost must be < {p*(1-p)/max(vg,1e-12):.1f} FPE per prefix", flush=True)

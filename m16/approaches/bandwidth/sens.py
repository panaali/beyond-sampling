"""H3/H5: sensitivity of the exact band offset delta to the other inputs.
(1) Structure function: S(eps) = E[(delta(y + eps*u) - delta(y))^2] / (2 Var delta), u a random unit direction in
    the 8 other coordinates (tie value v fixed), for eps from 1e-3 to 1. Smooth: S ~ eps^2; white noise: S ~ 1.
(2) Sobol total-effect indices: resample one input from its exact conditional law (max: N(0,1)|>v; lower:
    N(0,1)|<v) with everything else fixed; S_T(k) = E[(delta - delta_k)^2] / (2 Var delta). Also for v
    (resampled from its law, keeping it between the lower inputs and the max; rejected otherwise).
Lines whose perturbation breaks the rank-2-3 tie configuration are dropped (reported).
usage: python sens.py <model> [N]"""
import sys
import numpy as np
from scipy.special import ndtr, ndtri
import wtool as W

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
M = W.get_model(nm)
R = dict(np.load(f"data/{nm}.npz"))
keep = np.nonzero(R["v"] > 0)[0][:N]
y, a, b, p1, v = R["y"][keep], R["a"][keep], R["b"][keep], R["p1"][keep], R["v"][keep]
n = len(y); rows = np.arange(n)
CL = 0.3


def delta_of(yy):
    oth = yy.copy(); oth[rows, a] = -np.inf; oth[rows, b] = -np.inf
    s_ = -np.sort(-oth, 1); vv = yy[rows, a]
    ok = (s_[:, 0] > vv) & (s_[:, 1] < vv) & (vv > 0)
    tau = W.SQ2 * np.clip(np.minimum(s_[:, 0] - vv, vv - s_[:, 1]), 1e-6, None)
    out = W.walk(M, yy, a, b, tau)
    return np.clip(out["delta"], -CL, CL), out["mass"], ok


d0, m0, _ = delta_of(y)
V = d0.var(); Vm = m0.var()
rng = np.random.default_rng(7)
other = np.ones((n, 10), bool); other[rows, a] = False; other[rows, b] = False
print(f"{nm}: n={n} Var(delta clipped)={V:.3e} sd={np.sqrt(V):.4f}")
print("structure function S(eps) for delta / for mass (fraction of lines kept):")
res = {}
for eps in [1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    u = rng.standard_normal((n, 10)) * other; u /= np.linalg.norm(u, axis=1, keepdims=True)
    d1, m1, ok = delta_of(y + eps * u)
    S = np.mean((d1 - d0)[ok] ** 2) / (2 * V); Sm = np.mean((m1 - m0)[ok] ** 2) / (2 * Vm)
    res[eps] = (S, Sm)
    print(f"  eps={eps:<6} S_delta={S:.3f} S_mass={Sm:.3f} kept={ok.mean():.2f}")
# Sobol total effects, by role
print("Sobol total-effect (resample one input):")
ST = {}
roles = {}
order = np.argsort(np.argsort(-np.where(other, y, -np.inf), 1), 1)   # rank among others: 0 = max
for k in range(10):
    yk = y.copy()
    is_max = (k == p1)
    lo = ndtri(np.clip(rng.random(n) * ndtr(v), 1e-300, 1 - 1e-16))
    hi = ndtri(np.clip(ndtr(v) + rng.random(n) * (1 - ndtr(v)), 1e-300, 1 - 1e-16))
    yk[:, k] = np.where(is_max, hi, lo)
    sel = other[:, k]
    d1, m1, ok = delta_of(yk)
    ok &= sel
    # relative position of k w.r.t. a, b
    rel = np.where(k < a, "before a", np.where(k < b, "between", "after b"))
    for lab in ["before a", "between", "after b"]:
        for role, rm in [("max", is_max), ("lower", ~is_max)]:
            msk = ok & (rel == lab) & rm
            roles.setdefault((lab, role), []).append(((d1 - d0)[msk] ** 2, (m1 - m0)[msk] ** 2))
for key, lst in roles.items():
    dd = np.concatenate([x[0] for x in lst]); mm = np.concatenate([x[1] for x in lst])
    print(f"  {key[1]:5s} input {key[0]:9s}: S_T(delta)={dd.mean()/(2*V):.3f} S_T(mass)={mm.mean()/(2*Vm):.3f} (n={len(dd)})")
# total over all inputs, per line: sum of the 8 S_T (per line average)
# v (tie value) resample
vv = W.inv_ybar(rng.random(n)); yv = y.copy(); yv[rows, a] = vv; yv[rows, b] = vv
d1, m1, ok = delta_of(yv)
print(f"  tie value v: S_T(delta)={np.mean((d1-d0)[ok]**2)/(2*V):.3f} S_T(mass)={np.mean((m1-m0)[ok]**2)/(2*Vm):.3f} kept={ok.mean():.2f}")

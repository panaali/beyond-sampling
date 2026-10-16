"""Diagnosis: closed Sobol indices (relative to E Var(F|top)) of bath groups, for the exact 2-3 tie segment mass.
Groups by time relative to the top events (a, b, p1): before all / in between / after all; single inputs.
Closed index of group u: V_u = Cov(f(y), f(y^u)) - Cov(f(y), f(y^{none})), y^u keeps top+u, resamples other bath.
usage: python diag_bath2.py <model> [N]"""
import sys, time
import numpy as np
from scipy.special import ndtr, ndtri
import wtool as W

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
M = W.get_model(nm)
rng = np.random.default_rng(12)
y, a, b, p1, v, o1, o2 = W.surface(N, rng)
rows = np.arange(N)
low = np.ones((N, 10), bool); low[rows, a] = False; low[rows, b] = False; low[rows, p1] = False
lowidx = np.nonzero(low)[1].reshape(N, 7)
first = np.minimum(np.minimum(a, b), p1); last = np.maximum(np.maximum(a, b), p1)
grp = np.where(lowidx < first[:, None], 0, np.where(lowidx > last[:, None], 2, 1))   # (N,7)


def F(yy):
    oth = yy.copy(); oth[rows, a] = -np.inf; oth[rows, b] = -np.inf
    s_ = -np.sort(-oth, 1); vv = yy[rows, a]
    tau = W.SQ2 * np.clip(np.minimum(s_[:, 0] - vv, vv - s_[:, 1]), 1e-9, None)
    return W.walk(M, yy, a, b, tau)["mass"]


def newbath(keepmask):          # keepmask (N,7) bool: bath slots kept from y
    z = y.copy()
    vals = ndtri(np.clip(rng.random((N, 7)) * ndtr(v)[:, None], 1e-300, 1 - 1e-16))
    for j in range(7):
        z[rows, lowidx[:, j]] = np.where(keepmask[:, j], y[rows, lowidx[:, j]], vals[:, j])
    return z


t0 = time.time()
f0 = F(y); fn = F(newbath(np.zeros((N, 7), bool)))
base = np.mean(f0 * fn)
Vb = np.mean(f0 ** 2) - base           # E Var(F|top)
V = f0.var()
print(f"{nm}: N={N} CV {f0.std()/f0.mean():.2f}  bath share of Var(F): {Vb/V:.3f}   ({time.time()-t0:.0f}s)", flush=True)
for g, lab in enumerate(["before", "between", "after"]):
    fk = F(newbath(grp == g))
    print(f"  closed index of bath group '{lab}' / bath var: {(np.mean(f0*fk)-base)/Vb:.3f}  (mean group size {(grp==g).sum(1).mean():.2f})", flush=True)
tot1 = 0
for j in range(7):
    m = np.zeros((N, 7), bool); m[:, j] = True
    fk = F(newbath(m)); s = (np.mean(f0 * fk) - base) / Vb; tot1 += s
print(f"  sum of single-input first-order indices / bath var: {tot1:.3f}")
# rank-based: the largest bath input (o2) alone
rk = np.argsort(np.argsort(-y[rows[:, None], lowidx], 1), 1)
fk = F(newbath(rk == 0)); print(f"  closed index of the largest bath input (o2) / bath var: {(np.mean(f0*fk)-base)/Vb:.3f}")
fk = F(newbath(rk <= 1)); print(f"  closed index of the 2 largest bath inputs / bath var: {(np.mean(f0*fk)-base)/Vb:.3f}")
fk = F(newbath(rk <= 3)); print(f"  closed index of the 4 largest bath inputs / bath var: {(np.mean(f0*fk)-base)/Vb:.3f}")

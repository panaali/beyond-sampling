"""Diagnosis (random inputs allowed): variance decomposition of the exact 2-3 tie segment mass F.
top = (positions a,b,p1; tie value v; max o1); bath = the 7 lower inputs (iid N(0,1) | < v).
Reports: Var share of E[F|top]; bath total; sum of bath first-order effects (additivity); same for madj.
usage: python diag_bath.py <model> [N]"""
import sys
import numpy as np
from scipy.special import ndtr, ndtri
import wtool as W

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
M = W.get_model(nm)
rng = np.random.default_rng(11)
y, a, b, p1, v, o1, o2 = W.surface(N, rng)
rows = np.arange(N)
low = np.ones((N, 10), bool); low[rows, a] = False; low[rows, b] = False; low[rows, p1] = False
lowidx = np.nonzero(low)[1].reshape(N, 7)          # positions of bath inputs (sorted)


def F(yy):
    oth = yy.copy(); oth[rows, a] = -np.inf; oth[rows, b] = -np.inf
    s_ = -np.sort(-oth, 1); vv = yy[rows, a]
    tau = W.SQ2 * np.clip(np.minimum(s_[:, 0] - vv, vv - s_[:, 1]), 1e-9, None)
    out = W.walk(M, yy, a, b, tau)
    return out["mass"], out["madj"]


def newbath(yy, keep=None):
    z = yy.copy()
    u = rng.random((N, 7))
    vals = ndtri(np.clip(u * ndtr(v)[:, None], 1e-300, 1 - 1e-16))
    for j in range(7):
        if keep is not None and j == keep:
            continue
        z[rows, lowidx[:, j]] = vals[:, j]
    return z


f0, g0 = F(y)
fA, gA = F(newbath(y))                 # all bath new, top same
fB, gB = F(newbath(y))                 # independent second copy
for name, x0, xA, xB in [("mass", f0, fA, fB), ("madj", g0, gA, gB)]:
    V = x0.var()
    bath_tot = np.mean((x0 - xA) ** 2) / 2
    top = np.mean(x0 * xA) - x0.mean() * xA.mean()       # Var E[F|top]
    print(f"{nm} {name}: mean {x0.mean():.5f} CV {x0.std()/x0.mean():.2f}  Var(E[F|top])/V={top/V:.3f}  E Var(F|top)/V={bath_tot/V:.3f}")
S = []
for j in range(7):
    fk, gk = F(newbath(y, keep=j))
    # first-order bath effect of input j given top: E[f0 (fk - fA')], use fB as independent all-new copy
    S.append((np.mean(f0 * (fk - fB)) / f0.var(), np.mean(g0 * (gk - gB)) / g0.var()))
S = np.array(S)
print("bath first-order shares (mass) by bath slot j (sorted by position):", np.round(S[:, 0], 3), "sum", S[:, 0].sum().round(3))
print("bath first-order shares (madj):", np.round(S[:, 1], 3), "sum", S[:, 1].sum().round(3))

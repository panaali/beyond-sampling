"""DIAGNOSIS: first-order and total Sobol indices of delta (signed tie offset) w.r.t. each rest input,
given the top-3 configuration (Jansen pick-freeze estimators). Aggregated by where the rest input sits
relative to a < b and c."""
import numpy as np
from lp import *
from lp import _trunc_below
model = get_model("orig"); rng = np.random.default_rng(7)
ncfg, per = 40, 200
agg = {}
tot_first, tot_total = [], []
for j in range(ncfg):
    combo, u, xc, rest = rand23(1, rng)
    a, b, c = combo[0]
    oth = [i for i in range(T) if i not in (a, b, c)]
    C = np.repeat(combo, per, 0); U_ = np.repeat(u, per); X = np.repeat(xc, per)
    A = _trunc_below(U_[:, None], rng.random((per, 7))); Bm = _trunc_below(U_[:, None], rng.random((per, 7)))
    fA = np.clip(delta_ab(model, build23(C, U_, X, A))[0], -0.3, 0.3)
    fB = np.clip(delta_ab(model, build23(C, U_, X, Bm))[0], -0.3, 0.3)
    V = np.var(np.concatenate([fA, fB]))
    S1, ST = [], []
    for t in range(7):
        ABt = A.copy(); ABt[:, t] = Bm[:, t]
        fAB = np.clip(delta_ab(model, build23(C, U_, X, ABt))[0], -0.3, 0.3)
        S1.append(np.mean(fB * (fAB - fA)) / V); ST.append(0.5 * np.mean((fA - fAB) ** 2) / V)
        p = oth[t]
        cat = ("x9 " if p == 9 else "") + ("before a" if p < a else "a..b" if p < b else "after b") + (" after c" if p > c else "")
        agg.setdefault(cat, []).append((S1[-1], ST[-1]))
    tot_first.append(sum(S1)); tot_total.append(sum(ST))
print("sum of first-order indices: median %.2f; sum of total indices median %.2f" % (np.median(tot_first), np.median(tot_total)))
for k, v in sorted(agg.items()):
    v = np.array(v); print("%-22s n=%3d  S1 mean %.3f  ST mean %.3f" % (k, len(v), v[:, 0].mean(), v[:, 1].mean()))

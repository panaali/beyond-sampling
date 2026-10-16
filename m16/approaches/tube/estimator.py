"""Tube estimator: accuracy = 1 - [blind] - Z23 E[F23] - Z12 E[F12]  (README.md). Deterministic."""
import sys
sys.path.insert(0, "/home/user/clean-neurons/m16")
sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/tube")
import numpy as np
from tube import faces23, faces12, blind_points, F_faces, forward_from, P_BLIND, SHIFT, T

# budget shares (Neyman: n_i proportional to sd_i / sqrt(cost_i), measured on the ORIGINAL model only)
SHARE = dict(f23=0.66, f12=0.28, blind=0.06)
COST = dict(f23=8.9, f12=8.2)           # mean passes per face (original model), used only to size n
import os
_H = os.path.dirname(os.path.abspath(__file__))
Q23 = np.load(os.path.join(_H, "q23.npy")); Q12 = np.load(os.path.join(_H, "q12.npy"))


def est_tube(model, budget, rng=None, shift=SHIFT, share=SHARE, parts=False, imp=True):
    n23 = max(8, int(share["f23"] * budget / COST["f23"]))
    n12 = max(8, int(share["f12"] * budget / COST["f12"]))
    nb = max(11, int(share["blind"] * budget))
    f = faces23(n23, shift, Q23 if imp else None); F, _, C = F_faces(model, f, 400, radial=True)
    e23 = f["Z"] * (f["wt"] * F).mean(); c23 = C.sum()
    g = faces12(n12, shift, Q12 if imp else None); G, _, D = F_faces(model, g, 400, radial=True)
    e12 = g["Z"] * (g["wt"] * G).mean(); c12 = D.sum()
    xb = blind_points(nb, shift)
    eb = P_BLIND * float((forward_from(model, xb).argmax(1) != np.argsort(xb, 1)[:, -2]).mean())
    acc = 1 - eb - e23 - e12
    used = float(c23 + c12 + nb)
    if parts:
        return acc, used, dict(e23=e23, e12=e12, eb=eb, c23=c23 / n23, c12=c12 / n12)
    return float(acc), used


from tube import F_newton
SHARE_B = dict(n23=0.51, c23=0.21, n12=0.19, c12=0.065, blind=0.03)
COST_B = dict(n23=4.4, c23=9.0, n12=4.25, c12=8.2)


def est_tube_newton(model, budget, rng=None, shift=SHIFT, share=SHARE_B, parts=False, imp=True, J=3):
    """Newton crossing search (cheap, slightly biased: assumes one error interval per side starting at the tie)
    on n faces + exact cell walk on the first m of them as a correction (control-variate form)."""
    out, used = {}, 0.0
    for fam, faces, q in (("23", faces23, Q23), ("12", faces12, Q12)):
        n = max(16, int(share["n" + fam] * budget / COST_B["n" + fam]))
        m = max(4, min(n, int(share["c" + fam] * budget / COST_B["c" + fam])))
        f = faces(n, shift, q if imp else None)
        Fn, Cn = F_newton(model, f, J)
        sub = {k: (v[:m] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
        Fe, _, Ce = F_faces(model, sub, 400, radial=True)
        w = f["wt"]
        out["e" + fam] = f["Z"] * ((w * Fn).mean() + (w[:m] * (Fe - Fn[:m])).mean())
        used += Cn.sum() + Ce.sum()
    nb = max(11, int(share["blind"] * budget))
    xb = blind_points(nb, shift)
    eb = P_BLIND * float((forward_from(model, xb).argmax(1) != np.argsort(xb, 1)[:, -2]).mean())
    used += nb
    acc = 1 - eb - out["e23"] - out["e12"]
    if parts:
        return acc, used, dict(out, eb=eb)
    return float(acc), used

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

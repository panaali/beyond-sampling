"""Checks: (1) model-free face identity Z23 E[win] + Z12 E[win] = 1; (2) walk vs dense-grid brute force;
(3) estimator vs truth on small models. usage: python check_core.py"""
import sys, time
import numpy as np
from scipy.special import ndtr
import core as C

for T in (2, 3, 5, 10):
    _, Z23, Z12 = C.tables(T)
    tot = 0
    for kind, Z in (("23", Z23), ("12", Z12)):
        if Z == 0:
            continue
        U = C.sobol(T, 2 ** 16, 1)
        y, a, b, tau = C.face_points(T, kind, U)
        tot += Z * (2 * ndtr(tau) - 1).mean()
    print(f"T={T} identity sum = {tot:.5f} (should be 1)")

M = C.zoo(16, 10, seed=None)
T = 10
for kind in ("23", "12"):
    U = C.sobol(T, 64, 3)
    y, a, b, tau = C.face_points(T, kind, U)
    m, cost, cells = C.walk_mass(M, y, a, b, tau, kind)
    errs = []
    for i in range(30):
        tt = np.linspace(-tau[i], tau[i], 40001)
        d = np.zeros(T); d[a[i]] = 1 / C.SQ2; d[b[i]] = -1 / C.SQ2
        X = y[i] + tt[:, None] * d
        k = C.forward(M, X).argmax(1)
        tg = np.argsort(X, 1)[:, -2]
        w = (k != tg) * np.exp(-tt ** 2 / 2) / np.sqrt(2 * np.pi)
        errs.append(w.sum() * (tt[1] - tt[0]) - m[i])
    print(kind, "max |walk - grid|", np.abs(errs).max(), "mean mass", m.mean(), "cost/face", cost.mean(), "cells", cells.mean())

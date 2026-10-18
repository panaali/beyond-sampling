"""T330b: tier B v3, a rigorous "safe to tmax" screen with (1) the fixed-off fix (a gate proven off has deviation 0),
(2) a norm track nu_s >= ||dh_s(t') - dh*_s||_2 over the window, using per-gate-pattern sub-matrix norms
(min of Frobenius of Wh[N_s, A_{s-1}] and sigma_max(|Wh|) >= sigma_max(Wh)), and (3) per-neuron bounds
min(|Wh| e, ||Wh[i, A]||_2 nu). Readout bound: min(|Wo_k - Wo_j| . e, ||(Wo_k - Wo_j)[A_T]||_2 nu_T).
All of it sound: |Wh x|_i <= ||Wh[i, supp x]|| ||x||, ||Wh[N, A] x_A|| <= ||Wh[N, A]||_F ||x_A||.
Uses screen_c's walk (its `check` is replaced). README, T330b pre-registration."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import screen_c as SC  # noqa: E402
from cont_e import H, T, C  # noqa: E402

_cache = {}
F_SVD = 20_000                                   # sigma_max(|Wh|), once per model (charged by est)


def _pre(aWh):
    key = id(aWh)
    if key not in _cache:
        _cache.clear(); _cache[key] = (aWh * aWh, float(np.linalg.norm(aWh, 2)))
    return _cache[key]


def check(aWh, aWoDk, z, dz, g, m, dm, delta, s_root=None):
    W2, sig = _pre(aWh)
    e = np.zeros(H); nu = 0.0; f = 0.0; nz = False; A = np.ones(H, bool)
    for s in range(T):
        if nz:
            ez_box = aWh @ e
            rs = W2 @ A.astype(float)                       # squared row norms of Wh[:, A]
            ez = np.minimum(ez_box, np.sqrt(rs) * nu)
            f += 2 * H * H + 2 * H * H + 3 * H
        else:
            ez = np.zeros(H); rs = None; f += 0
        f += 4 * H
        zs = z[s]
        if s_root is not None and s == s_root[0]:
            zs = zs.copy(); zs[s_root[1]] = 0.0
        lo, hi = dz[s] - ez, dz[s] + ez
        zlo = zs + delta * np.minimum(0.0, lo); zhi = zs + delta * np.maximum(0.0, hi)
        fixed = np.where(g[s], zlo >= 0.0, zhi <= 0.0)
        fixed_off = fixed & ~g[s]; free = ~fixed
        e = np.where(fixed_off, 0.0, np.where(fixed, ez, np.abs(dz[s]) + ez))
        if nz:
            Nset = ~fixed_off
            nz_norm = min(float(np.sqrt(rs[Nset].sum())), sig) * nu      # ||dz deviation on N||
            nu = min(float(np.sqrt((e * e).sum())), nz_norm + float(np.sqrt((dz[s][free] ** 2).sum())))
            f += 6 * H
        else:
            nu = float(np.sqrt((e * e).sum())); f += 2 * H
        A = ~fixed_off
        nz = nz or bool(e.any())
    box = aWoDk @ e
    nrm = np.sqrt((aWoDk * aWoDk) @ A.astype(float)) * nu
    bound = m + delta * np.minimum(0.0, dm - np.minimum(box, nrm)); f += 2 * C * H + 2 * C * H + 6 * C
    return bool(bound.min() > 0), f


def est(model, **kw):
    orig = SC.check; SC.check = check
    try:
        out = SC.est(model, **kw)
    finally:
        SC.check = orig
    if kw.get("parts"):
        e, fl, info, prof = out
        return e, fl + (F_SVD if kw.get("lines", ("nb-23", "nb-12")) else 0), info, prof
    return out[0], out[1] + (F_SVD if kw.get("lines", ("nb-23", "nb-12")) else 0)

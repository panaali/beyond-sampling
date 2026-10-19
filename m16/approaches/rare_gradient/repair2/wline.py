"""T411 design tool: line-CMC along v1 for W(t0) = {argmax != p2, g23 > t0} (SAMPLING-CLASS, unbiased).
Given the rest of a draw from the law given g23 > t0, v1 has density phi on (v2, inf); per line the wrong mass is
1 - mass{argmax = p2} / (1 - Phi(v2)); dP_W/dW = -(boundary terms of the correct set). w(model, t0, n, seed, grad)."""
import os
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repair"))
import lcmc  # noqa: E402
import truthc_copy as TCC  # noqa: E402

VMAX = 9.0


def w(model, t0, n, seed, grad=True, chunk=20_000, U=None):
    f = TCC.dens(t0); c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(TCC.GRID))]); c /= c[-1]
    pg = TCC.p_gap(t0); rng = np.random.default_rng(seed); sZ = sZ2 = 0.0; sg = sg2 = None; fl = 0.0; m0 = 0; sp = sp2 = 0.0
    while m0 < n:
        m = min(chunk, n - m0); x, perm, v2, _ = lcmc._draw_e1(c, t0, rng, m); p1, p2 = perm[:, 0], perm[:, 1]; r = np.arange(m)
        d = np.zeros_like(x); d[r, p1] = 1.0; x0 = x.copy(); x0[r, p1] = 0.0
        lo, hi = v2.copy(), np.maximum(v2 + 1e-9, VMAX); nrm = ndtr(-lo) - ndtr(-hi)
        Zc, G, f_ = lcmc._accumulate(model, x0, d, p2, lo, hi, nrm, r, m, grad); fl += f_
        Z = 1.0 - Zc; sZ += Z.sum(); sZ2 += (Z ** 2).sum()
        if grad:
            G = -G
            if U is not None:
                pr = G @ U.T; sp = sp + pr.sum(0); sp2 = sp2 + (pr ** 2).sum(0)
            sg = G.sum(0) if sg is None else sg + G.sum(0); sg2 = (G ** 2).sum(0) if sg2 is None else sg2 + (G ** 2).sum(0)
        m0 += m
    out = lcmc._finish(sZ, sZ2, sg, sg2, n, pg, fl)
    if U is not None:
        out.update(proj=pg * sp / n, proj_se=pg * np.sqrt(np.maximum(sp2 / n - (sp / n) ** 2, 0) / n))
    return out

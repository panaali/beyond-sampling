"""T351 / T317h: cont_g's estimator (T317f with T339's low pattern; spread 0 = T317f bitwise) plus T347's tilt
correction of the line-set product's overstatement (frozen formula t347/tilt.py @790de13d), with the declared changes:
walks not repeated (theta* from the estimator's own merged nb intervals), crossing slopes by one value pass + 2 tangents
at each wall crossing (charged as executed), 8-point Gauss-Legendre g-rule. Sampling-free; no fitted constant.
est_h(model, spread, tilt) -> (estimate, FLOPs as executed)."""
import os
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont_g as CG  # noqa: E402

H, T, C, TMAX, NT, TRI = CG.H, CG.T, CG.C, CG.TMAX, CG.NT, CG.TRI
NQ = 8
_q, _w = np.polynomial.legendre.leggauss(NQ); GQ = (_q + 1) / 2 * TMAX; WQ = _w * TMAX / 2
F_CROSS = T * ((2 * H * H + 2 * H) + 2 * H + 2 * CG.F_TAN) + 3 * 2 * C * H + 6 * C + 4      # = 17,984 per wall row
F_D_ROW = NQ * 5 + CG.G * (NQ * 27 + 26)                                                     # = 4,880 per wall row
TABLES = {s: CG._tables(s) for s in (0.0, -1.0)}                                             # model-free, uncharged
_nb = TABLES[0.0][0]; V = _nb["nodes"]
W23 = CG._pdf(V[:, None] + GQ[None])                                   # phi(v2 + g12): model-free table (G, NQ)
W12 = 8 * CG._pdf(V[:, None] - GQ[None]) * ndtr(V[:, None] - GQ[None]) ** 7   # density of g23 given v2 (G, NQ)
P8V = ndtr(V) ** 8; PV = ndtr(V)


def _first_start(R, A, n):
    th = np.full(n, TMAX); np.minimum.at(th, R, np.where(A < TMAX, A, TMAX)); return th


def _cross(model, x, dx, cross, k):
    """Value pass + 2 tangents (gates frozen) at x: binding-rival margin slopes along dx (alpha) and cross (beta)."""
    Wi, Wh, Wo = model; n = len(x); h = np.zeros((n, H)); da = np.zeros_like(h); db = np.zeros_like(h)
    for s in range(T):
        z = h @ Wh.T + x[:, s:s + 1] * Wi[None]; g = z > 0
        da = np.where(g, da @ Wh.T + dx[:, s:s + 1] * Wi[None], 0.0); db = np.where(g, db @ Wh.T + cross[:, s:s + 1] * Wi[None], 0.0)
        h = np.where(g, z, 0.0)
    L, La, Lb = h @ Wo.T, da @ Wo.T, db @ Wo.T
    rr = np.arange(n); M = L[rr, k][:, None] - L; M[rr, k] = np.inf; riv = M.argmin(1)
    return La[rr, k] - La[rr, riv], Lb[rr, k] - Lb[rr, riv]


def est_h(model, spread=0.0, tilt=False, parts=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = TRI[:, 1]; flops = 0.0; tot = np.zeros(NT); keep = {}
    tabs = TABLES[spread] if spread in TABLES else CG._tables(spread)
    for t in tabs:                                           # cont_g.est, verbatim logic
        nodes = t["nodes"]; I = {}
        for kind in ("23", "12"):
            if f"line{kind}" not in t:
                continue
            x, dx = t[f"line{kind}"]
            p, f = CG.walk_c1(model, x, dx, k, TMAX); flops += f
            R, A, B, ops = CG.merge(p, NT, TMAX); flops += ops
            I[kind], f = CG.profile(R, A, B, NT, nodes, kind, TMAX, t["phi0"]); flops += f
            if t["name"] == "nb":
                keep[kind] = (R, A)
        I12 = I["12"] if "12" in I else np.repeat(t["blind12"][None], NT, 0)
        tot += (I12 * I["23"] * t["dens"][None]) @ t["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    acc = float(tot.mean()); D = 0.0; info = {}
    if tilt:
        nb = tabs[0]; (x23, d23), (x12, d12) = nb["line23"], nb["line12"]; v2s, g12s, g23s = nb["point"]
        Dt = np.zeros((NT, len(V)))
        for kind, x, dx, cross, gs, W, Psi in (
                ("23", x23, d23, d12, g12s, W23, lambda c: P8V[None, :, None] - ndtr(V[None, :, None] - c) ** 8),
                ("12", x12, d12, d23, g23s, W12, lambda c: ndtr(V[None, :, None] + c) - PV[None, :, None])):
            R, A = keep[kind]; th = _first_start(R, A, NT); flops += 2 * len(R)
            w = np.flatnonzero((th > 0) & (th < TMAX))
            if not len(w):
                continue
            a, b = _cross(model, x[w] + (th[w] + 1e-9)[:, None] * dx[w], dx[w], cross[w], k[w]); flops += len(w) * F_CROSS
            s = np.where(np.abs(a) > 1e-12, -b / np.where(np.abs(a) > 1e-12, a, 1.0), 0.0)
            thg = np.clip(th[w, None, None] + s[:, None, None] * (GQ[None, None] - gs), 0, TMAX)       # (w, 1, NQ)
            Dt[w] += ((W[None] * (Psi(thg) - Psi(th[w, None, None]))) @ WQ)
            flops += len(w) * F_D_ROW
            info[f"walls{kind}"] = len(w)
        corr = (Dt * nb["dens"][None]) @ nb["wts"]; flops += NT * len(V) * 3
        D = float(corr.mean()); acc -= D
    return (acc, flops, dict(D=D, **info)) if parts else (acc, flops)

"""Shared helpers for sci_laplace (face sampling for DIAGNOSIS, exact walks reused from ../tube/tube.py)."""
import sys
sys.path.insert(0, "/home/user/clean-neurons/m16")
sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/tube")
import numpy as np
from scipy.special import ndtr, ndtri
import tube as TB
from tube import walk_side, W, T, COMBO23, CDF23, _inv, _trunc_above, _trunc_below

COMBO = np.array(COMBO23)


def get_model(name):
    if name == "orig":
        from common import Wi, Wh, Wo
        return (Wi.copy(), Wh.copy(), Wo.copy())
    if name.startswith("seed"):
        from alg_zoo import zoo_2nd_argmax
        m = zoo_2nd_argmax(16, 10, seed=int(name[4:]))
        f = lambda t: t.detach().numpy().astype(np.float64)
        return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)
    if name.startswith("s0."):
        from arc_eval import noisy_models
        sig, k = name[1:].split("_")
        return noisy_models(6, float(sig), seed=0)[int(k)]
    raise ValueError(name)


def build23(combo, u, xc, rest):
    """combo (n,3) = (a,b,c) with a<b; rest (n,7) values for the other positions in increasing index order."""
    n = len(u); x0 = np.empty((n, T)); d = np.zeros((n, T))
    for k in range(n):
        a, b, c = combo[k]
        oth = [i for i in range(T) if i not in (a, b, c)]
        x0[k, a] = x0[k, b] = u[k]; x0[k, c] = xc[k]; x0[k, oth] = rest[k]
        d[k, a], d[k, b] = 0.5, -0.5
    v4 = rest.max(1)
    S = np.minimum(2 * (xc - u) / 3, 2 * (u - v4))
    lo = np.maximum(0, -2 * u)
    return dict(x0=x0, d=d, lo=lo, hi=S, tpos=combo[:, 0], tneg=combo[:, 1])


def rand23(n, rng, combo=None):
    if combo is None:
        combo = COMBO[rng.integers(0, 360, n)]
    U = rng.random((n, 9))
    u = _inv(CDF23, U[:, 0]); xc = _trunc_above(u, U[:, 1]); rest = _trunc_below(u[:, None], U[:, 2:9])
    return combo, u, xc, rest


def F_sides(model, f, K=400):
    """Error mass (weight exp(-s^2/4), s = x_a - x_b) on each side; no radial integration."""
    e1, r1, c1 = walk_side(model, f["x0"], f["d"], f["lo"], f["hi"], f["tpos"], K)
    e2, r2, c2 = walk_side(model, f["x0"], -f["d"], f["lo"], f["hi"], f["tneg"], K)
    return e1, e2, c1 + c2


def fwd_jvp(model, x, dx):
    return TB.fwd_jvp(model, x, dx)


def delta_ab(model, f, K=60):
    """Signed network tie offset along the segment: the s (|s| < hi) nearest to 0 on the side the network
    gets wrong, where D(s) = L_a - L_b crosses 0.  Returns delta (0 if no crossing side; +-hi if D never
    crosses inside the segment), the tie margin D(0), and cells walked."""
    x0, d, hi = f["x0"], f["d"], f["hi"]; n = len(x0); a, b = f["tpos"], f["tneg"]; ar = np.arange(n)
    L, dL, P, dP = fwd_jvp(model, x0 + 1e-9 * d, d)
    D0 = L[ar, a] - L[ar, b]
    sg = np.where(D0 > 0, -1.0, 1.0)            # direction where the network is wrong near the tie
    dd = d * sg[:, None]
    delta = np.zeros(n); r = np.zeros(n); act = np.ones(n, bool); cells = np.zeros(n)
    for _ in range(K):
        ii = np.where(act)[0]
        if len(ii) == 0:
            break
        re = r[ii] + 1e-10
        L, dL, P, dP = fwd_jvp(model, x0[ii] + re[:, None] * dd[ii], dd[ii]); cells[ii] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            root = -P / dP
        root[~(root > 0)] = np.inf
        rn = np.minimum(re + root.reshape(len(ii), -1).min(1), hi[ii])
        Dv = (L[np.arange(len(ii)), a[ii]] - L[np.arange(len(ii)), b[ii]]) * sg[ii]   # want it to become >= 0
        dD = (dL[np.arange(len(ii)), a[ii]] - dL[np.arange(len(ii)), b[ii]]) * sg[ii]
        with np.errstate(divide="ignore", invalid="ignore"):
            rz = re - Dv / dD
        hit = (Dv >= 0) | ((dD > 0) & (rz <= rn))
        rz = np.where(Dv >= 0, re, rz)
        done = hit | (rn >= hi[ii] - 1e-12)
        delta[ii[done]] = np.where(hit[done], rz[done], hi[ii[done]]) * sg[ii[done]]
        act[ii[done]] = False
        r[ii[~done]] = rn[~done]
    delta[act] = r[act] * sg[act]
    return delta, D0, cells

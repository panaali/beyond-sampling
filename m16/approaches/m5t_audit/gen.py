"""T417: adversarial simplex generators for the m5t_fast audit. Each case is a dict with the exact rational vertices
(on the slice m_p2 = 1 + m_p3, checked exactly), the float enclosure given to fast, the exact volume, the tangent
centre mc (given to both fast and the reference) and the z-bin parameters."""
import math
import os

import numpy as np
from flint import fmpq, fmpq_mat
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ["real", "child", "shrink", "coinc", "flat", "needle", "huge", "far", "zbins", "under", "encl", "synth", "advmc"]


def Q(x):
    return fmpq(*float(x).as_integer_ratio())


class Seeds:
    def __init__(self):
        g = np.load(os.path.join(HERE, "data/geom.npz"))
        self.cells = []
        for j in range(int(g["n"])):
            d, mlow, jp3, jp2, k, i, reg = (int(v) for v in g[f"c{j}_meta"])
            self.cells.append(dict(Mv=g[f"c{j}_Mlo"], S=g[f"c{j}_S"], d=d, mlow=mlow, jp3=jp3, jp2=jp2, region=bool(reg)))

    def simplex(self, rng):
        c = self.cells[rng.integers(len(self.cells))]
        s = c["S"][rng.integers(len(c["S"]))]
        return c["Mv"][s].copy(), c


def fix_slice(M, jp2, jp3):
    """Make every vertex satisfy m_p2 = 1 + m_p3 exactly (in floats); returns None if it cannot."""
    M = M.copy()
    for v in range(M.shape[0]):
        y = M[v, jp3]
        for _ in range(3):
            p2 = 1.0 + y
            if Q(p2) == 1 + Q(y):
                break
            y = p2 - 1.0
        else:
            return None
        if not (np.isfinite(p2) and np.isfinite(y)):
            return None
        M[v, jp3] = y; M[v, jp2] = p2
    return M


def volume(M, jp2):
    """Exact |det| / n! in the coordinates other than p2 (the slice parametrisation)."""
    V, d = M.shape
    n = d - 1
    cols = [j for j in range(d) if j != jp2]
    rows = [Q(M[i, j]) - Q(M[0, j]) for i in range(1, V) for j in cols]
    return abs(fmpq_mat(n, n, rows).det()) / math.factorial(n)


def bisect(M, rng, longest=True):
    V = M.shape[0]
    if longest:
        best = (-1, 0, 1)
        for i in range(V):
            for j in range(i + 1, V):
                e = float(((M[i] - M[j]) ** 2).sum())
                if e > best[0]:
                    best = (e, i, j)
        _, i, j = best
    else:
        i, j = rng.choice(V, 2, replace=False)
    mid = 0.5 * (M[i] + M[j])
    M = M.copy()
    M[i if rng.random() < 0.5 else j] = mid
    return M


def pick_mc(M, rng, mode):
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
    import m5t_fast as F
    V, d = M.shape
    if mode == "wolfe":                                  # exactly what bounds_batch does when mc is None
        Mf = M[None]
        return np.einsum("nv,nvd->nd", F.min_norm_weights(Mf), Mf)[0]
    if mode == "nnls":
        rho = 1e3
        b, _ = nnls(np.vstack([M.T, rho * np.ones(V)]), np.r_[np.zeros(d), rho])
        b = b / b.sum() if b.sum() > 0 else np.full(V, 1.0 / V)
        return b @ M
    if mode == "zero":
        return np.zeros(d)
    if mode == "vertex":
        return M[rng.integers(V)].copy()
    if mode == "noise":
        return M.mean(0) + rng.standard_normal(d) * 10 ** rng.uniform(-3, 1.5)
    if mode == "far":
        return rng.standard_normal(d) * 10 ** rng.uniform(0, 3)
    if mode == "tiny":
        return rng.standard_normal(d) * 10 ** rng.uniform(-300, -100)
    raise ValueError(mode)


def make(family, rng, seeds):
    """One adversarial case, or None if the construction failed (counted, not tested)."""
    zb = dict(t0=None, zbins=24, zratio=fmpq(23, 20))
    if family == "synth":
        d = int(rng.integers(2, 13)); mlow = int(rng.choice([0, 1, 2, 5, 8]))
        jp2, jp3 = (int(v) for v in rng.choice(d, 2, replace=False))
        M = rng.standard_normal((d, d)) * 10 ** rng.uniform(-2, 1)
        sgn = rng.integers(3)
        if sgn == 0:
            M[:, jp3] = np.abs(M[:, jp3])
        elif sgn == 1:
            M[:, jp3] = -np.abs(M[:, jp3])
        cell = dict(d=d, mlow=mlow, jp3=jp3, jp2=jp2, region=False)
        zb["t0"] = float(rng.uniform(0.1, 5))
    else:
        M, cell = seeds.simplex(rng)
        d, mlow, jp3, jp2 = cell["d"], cell["mlow"], cell["jp3"], cell["jp2"]
        zb["t0"] = 1.5 if cell["region"] else float(rng.choice([2.25, 2.5]))
    V = M.shape[0]
    cen = M.mean(0)
    if family in ("child", "zbins", "encl", "advmc", "coinc") or (family == "under" and rng.random() < 0.5):
        for _ in range(int(rng.integers(0, 15))):
            M = bisect(M, rng, longest=rng.random() < 0.7)
        cen = M.mean(0)
    if family == "shrink":
        c = cen if rng.random() < 0.5 else M[rng.integers(V)]
        M = c + 10 ** -rng.uniform(1, 15.5) * (M - c)
    elif family == "coinc":
        for _ in range(int(rng.integers(1, 4))):
            a, b = rng.choice(V, 2, replace=False)
            cols = rng.choice(d, int(rng.integers(1, d + 1)), replace=False)
            M[a, cols] = M[b, cols]                    # shared coordinates (incl. p3 / whole vertices): tied knots
        if rng.random() < 0.3:
            M = cen + np.round((M - cen) * 8) / 8      # coarse grid: many exact ties
    elif family == "flat":
        w = rng.dirichlet(np.ones(V - 1)); i = int(rng.integers(V))
        others = np.delete(M, i, 0); p = w @ others
        M[i] = p + 10 ** -rng.uniform(1, 15) * (M[i] - p)
    elif family == "needle":
        i = int(rng.integers(V)); c = M[[j for j in range(V) if j != i]].mean(0)
        e = 10 ** -rng.uniform(1, 12)
        M = np.where(np.arange(V)[:, None] == i, M, c + e * (M - c))
    elif family == "huge":
        M = cen + 10 ** rng.uniform(0.3, 3) * (M - cen)
    elif family in ("far", "under"):
        sh = rng.standard_normal(d); sh[jp2] = 0
        M = M + sh / np.linalg.norm(sh) * 10 ** rng.uniform(0, 2.5 if family == "far" else 3)
        if family == "under":
            zb["t0"] = float(10 ** rng.uniform(0.5, 2))
    if family == "zbins":
        zb["t0"] = float(rng.choice([1e-3, 0.05, 0.5, 1.5, 2.5, 6, 15, 40]))
        zb["zratio"] = [fmpq(2 ** 20 + 1, 2 ** 20), fmpq(10001, 10000), fmpq(101, 100), fmpq(23, 20), fmpq(3, 2), fmpq(3), fmpq(20)][rng.integers(7)]
        zb["zbins"] = int(rng.choice([2, 3, 8, 24, 48]))
    M = fix_slice(M, jp2, jp3)
    if M is None or not np.all(np.isfinite(M)):
        return None
    vol = volume(M, jp2)
    Mlo = M.copy(); Mhi = M.copy()
    if family == "encl":
        r = 10 ** -rng.uniform(0, 16) * (np.abs(M) + 10 ** -rng.uniform(0, 30))
        Mlo = M - r * rng.random(M.shape); Mhi = M + r * rng.random(M.shape)
    mode = "wolfe" if rng.random() < 0.5 else "nnls"
    if family == "advmc":
        mode = str(rng.choice(["zero", "vertex", "noise", "far", "tiny", "wolfe"]))
    try:
        mc = pick_mc(M, rng, mode)
    except Exception:  # noqa: BLE001
        return None
    mc = np.asarray(mc, float)                           # a non-finite Wolfe centre is kept: fast must not return NaN
    from m5t_fast import q2iv
    vlo, vhi = q2iv(vol) if vol != 0 else (0.0, 0.0)
    return dict(family=family, d=d, mlow=mlow, jp3=jp3, jp2=jp2, M=M, Mlo=Mlo, Mhi=Mhi, vol=vol, vlo=vlo, vhi=vhi,
                mc=np.asarray(mc, float), mcmode=mode, **zb)

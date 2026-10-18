"""Shared pieces for the M_4,3 reproduction (T36): weights, ARC's noisy copies, forward pass, MC truth,
spherical-polygon geometry (exact cone volumes in R^3) and an operation counter.

Operation convention (ARC, Hilton notes Dec 2024): one multiply-add = 1 op, so one forward pass of M_4,3 is
3*4*(4+1) + 4*3 = 72 ops. Our leaderboard counts FLOPs (mult and add separately, 144 per pass); ratios are the
same in either convention as long as both sides use it.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_w = np.load(os.path.join(HERE, "m43_weights.npz"))
Wi0, Wh0, Wo0 = _w["Wi"], _w["Wh"], _w["Wo"]      # (4,), (4,4), (3,4): alg_zoo.zoo_2nd_argmax(4,3) (seed 0)
T, H = 3, 4
OPS_PER_SAMPLE = 3 * 4 * (4 + 1) + 4 * 3           # 72, ARC's count


def noisy_models(k=256, sigma=0.05, seed=0):
    """ARC protocol: zero-mean Gaussian noise, absolute s.d. sigma, on every weight (32 parameters)."""
    rng = np.random.default_rng(seed)
    return [(Wi0 + sigma * rng.standard_normal(Wi0.shape),
             Wh0 + sigma * rng.standard_normal(Wh0.shape),
             Wo0 + sigma * rng.standard_normal(Wo0.shape)) for _ in range(k)]


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def mc_acc(model, n, rng, chunk=1_000_000):
    ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done)
        x = rng.standard_normal((m, T))
        ok += (forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum(); done += m
    return ok / n


class Ops:
    """Operation counter (multiply-add = 1 op; a sqrt, division or atan2 = 1 op)."""
    def __init__(self):
        self.n = 0

    def add(self, k):
        self.n += k


# ---------------------------------------------------------------- spherical polygons (convex, pointed cones)
def simplicial(N):
    """Rays (cyclic order) of the cone {x : N x >= 0}, N 3x3 invertible."""
    R = np.linalg.inv(N)                       # columns r_i with N r_i = e_i
    return [R[:, i] / np.linalg.norm(R[:, i]) for i in range(3)]


def clip(poly, n, ops=None):
    """Intersect the convex spherical polygon `poly` (list of unit rays, cyclic) with {x : n.x >= 0}.
    Returns (inside, outside) polygons (either may be [] = measure zero)."""
    s = [float(n @ v) for v in poly]
    if ops is not None:
        ops.add(3 * len(poly))
    if min(s) >= 0:
        return poly, []
    if max(s) <= 0:
        return [], poly
    ins, out = [], []
    m = len(poly)
    for i in range(m):
        j = (i + 1) % m
        vi, si, sj = poly[i], s[i], s[j]
        if si >= 0:
            ins.append(vi)
        if si <= 0:
            out.append(vi)
        if (si > 0 and sj < 0) or (si < 0 and sj > 0):
            p = abs(si) * poly[j] + abs(sj) * vi
            p = p / np.linalg.norm(p)
            if ops is not None:
                ops.add(6 + 4)
            ins.append(p); out.append(p)
    return (ins if len(ins) >= 3 else []), (out if len(out) >= 3 else [])


def frac(poly, ops=None):
    """Solid angle of a convex spherical polygon / (4 pi) = standard-Gaussian measure of the cone.
    Fan triangulation, Van Oosterom-Strackee formula for each triangle (equivalent to Girard's excess)."""
    if len(poly) < 3:
        return 0.0
    a = poly[0]; tot = 0.0
    for i in range(1, len(poly) - 1):
        b, c = poly[i], poly[i + 1]
        num = abs(a @ np.cross(b, c))
        den = 1.0 + a @ b + a @ c + b @ c
        tot += 2.0 * np.arctan2(num, den)
        if ops is not None:
            ops.add(9 + 9 + 3)                 # triple product, three dots + adds, atan2
    return tot / (4 * np.pi)


def cone_frac(normals, ops=None):
    """Measure of {x : n_i.x >= 0 for all i}; the first three normals must be independent (pointed start)."""
    poly = simplicial(np.array(normals[:3]))
    for n in normals[3:]:
        poly, _ = clip(poly, np.asarray(n), ops)
        if not poly:
            return 0.0
    return frac(poly, ops)


# orderings: perm = (a, b, c) means x_a >= x_b >= x_c; the 2nd argmax is b
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def order_normals(perm):
    a, b, c = perm
    n1 = np.zeros(3); n1[a] = 1; n1[b] = -1
    n2 = np.zeros(3); n2[b] = 1; n2[c] = -1
    return [n1, n2]

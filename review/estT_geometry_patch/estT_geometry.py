"""T316 proposed estT geometry patch (drop-in; does NOT edit any shared estT copy).

Defect (T270, found by the T192 child, cx6.py): for some non-simplicial k = 6 cells the stored rays c.R miss true vertices
(clip's combinatorial adjacency test misses edges of degenerate cells), and the pulling triangulation over the symbolic
incidence then drops more volume -> cell masses are underestimates.

Fix (same method as T270's remass6.py, packaged as estT's mass()): the H-representation c.N is authoritative. For a
non-simplicial cell, enumerate its rays independently from c.N (qhull HalfspaceIntersection on the slice x . c = 1,
c = normalised sum of unit inward normals, interior point = Chebyshev centre by LP), triangulate the slice points by
Delaunay, and feed each simplex to the unchanged simplicial routine. Simplicial cells (len(R) == d) are untouched.
If qhull fails (no interior point at LP radius <= RMIN, or a qhull precision error), the original routine is used and
the failure is counted, so the patch never does worse than estT on cells it cannot handle.

API
  simplices(c) -> (s, d, d) unit-ray simplices, or None on failure
  make_mass(orig_mass, sink) -> mass(c, ops) drop-in; sink(R) receives each simplex (estT: orthant; estG: BUF push)
  apply_estT(B)        B.mass <- patched (orthant sums), returns the original
  apply_estG(estG, B)  B.mass / estG._tri_mass <- patched (simplices pushed to estG.BUF[0]), returns the original
  STATS                dict(cells_qhull, cells_fallback, cells_simplicial)
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay, HalfspaceIntersection

RMIN = 1e-12
STATS = dict(cells_qhull=0, cells_fallback=0, cells_simplicial=0)


def simplices(c):
    N = c.N / np.linalg.norm(c.N, axis=1, keepdims=True); d = N.shape[1]
    cdir = N.sum(0); cdir /= np.linalg.norm(cdir)
    U = np.linalg.svd(np.eye(d) - np.outer(cdir, cdir))[0][:, :d - 1]
    A = -(N @ U); b = -(N @ cdir); nrm = np.linalg.norm(A, axis=1)                  # slice: A z + b <= 0
    lp = linprog(np.r_[np.zeros(d - 1), -1], A_ub=np.c_[A, nrm], b_ub=-b, bounds=[(None, None)] * (d - 1) + [(0, None)])
    if lp.status != 0 or lp.x[-1] <= RMIN:
        return None
    Z = HalfspaceIntersection(np.c_[A, b], lp.x[:d - 1]).intersections
    R = cdir + Z @ U.T; R /= np.linalg.norm(R, axis=1, keepdims=True)
    R = np.unique(np.round(R, 12), axis=0)
    if len(R) < d:
        return None
    Zs = (R / (R @ cdir)[:, None] - cdir) @ U
    S = R[Delaunay(Zs).simplices]
    return S / np.linalg.norm(S, axis=2, keepdims=True)


def make_mass(orig_mass, sink):
    def mass(c, ops):
        d = c.R.shape[1]
        if len(c.R) == d:
            STATS["cells_simplicial"] += 1
            return orig_mass(c, ops)
        try:
            S = simplices(c)
        except Exception:  # noqa: BLE001  qhull precision errors on extreme cells
            S = None
        if S is None:
            STATS["cells_fallback"] += 1
            return orig_mass(c, ops)
        STATS["cells_qhull"] += 1
        ops.add(len(c.N) * d * 40 + len(S) * d ** 3)     # nominal geometry charge (LP + qhull + Delaunay), see README
        return float(sum(sink(s, ops) for s in S))
    return mass


def apply_estT(B):
    orig = B.mass
    B.mass = make_mass(orig, lambda s, ops: B.orthant(s, ops))
    return orig


def apply_estG(estG, B):
    orig = B.mass

    def push(s, ops):
        estG.BUF[0].add(s); return 0.0
    B.mass = estG._tri_mass = make_mass(orig, push)
    return orig


# ------------------------------------------------------------------ part B: clip on a repaired V-representation (optional)
TOL_INC = 1e-10
STATS_B = dict(clip_checked=0, clip_rebuilt=0, clip_qhull_fail=0)


def rays(c):
    """independent unit rays of {N x >= 0} (qhull on the slice), or None"""
    S = simplices(c)
    if S is None:
        return None
    return np.unique(np.round(S.reshape(-1, S.shape[-1]), 12), axis=0)


def rebuild(c, R):
    """Cone with rays R and NUMERIC incidence (|n_i . r| <= TOL_INC), same normals"""
    Nu = c.N / np.linalg.norm(c.N, axis=1, keepdims=True)
    return type(c)(R, c.N.copy(), np.abs(R @ Nu.T) <= TOL_INC)


def apply_clip(B):
    """B.clip <- check the cell's rays against qhull first; if the counts differ, clip the rebuilt cell instead"""
    orig = B.clip

    def clip(c, n, ops):
        d = c.R.shape[1]
        if len(c.R) > d:
            STATS_B["clip_checked"] += 1
            try:
                R = rays(c)
            except Exception:  # noqa: BLE001
                R = None
            if R is None:
                STATS_B["clip_qhull_fail"] += 1
            elif len(R) != len(c.R):
                STATS_B["clip_rebuilt"] += 1; c = rebuild(c, R)
        return orig(c, n, ops)
    B.clip = clip
    return orig

"""T192: numba versions of the two geometry hot spots of estT (85% of (8,5) geometry time):
estT._adjacent_pairs (edges of a cone from its symbolic incidence) and mass.faces_of (pulling triangulation).
Same loops, same order, same op charges; matrix_rank = count(sv > sv.max() * max(shape) * eps) as numpy.
Importing patches estT_base._adjacent_pairs and estT_base.mass (after estG)."""
import numpy as np
from numba import njit

import estG
import estT_base as B

EPS = np.finfo(float).eps


@njit(cache=True)
def _rank(M):
    r, c = M.shape
    if r == 0:
        return 0
    s = np.linalg.svd(M)[1]
    tol = s.max() * max(r, c) * EPS
    n = 0
    for v in s:
        if v > tol:
            n += 1
    return n


@njit(cache=True)
def _adj(Tm, N, d):
    m, f = Tm.shape
    out = np.empty((m * (m - 1) // 2, 2), np.int64); no = 0; ops = 0
    for a in range(m):
        for b in range(a + 1, m):
            Z = Tm[a] & Tm[b]; ops += f
            zs = 0
            for i in range(f):
                if Z[i]:
                    zs += 1
            if zs < d - 2:
                continue
            ops += (m - 2) * zs
            found = False
            for q in range(m):
                if q == a or q == b:
                    continue
                allz = True
                for i in range(f):
                    if Z[i] and not Tm[q, i]:
                        allz = False
                        break
                if allz:
                    found = True
                    break
            if found:
                continue
            if _rank(N[Z]) >= d - 2:
                ops += d ** 3; out[no, 0] = a; out[no, 1] = b; no += 1
    return out[:no], ops


def adjacent_pairs(c, ops):
    pr, o = _adj(np.ascontiguousarray(c.T), np.ascontiguousarray(c.N), c.R.shape[1])
    ops.add(int(o))
    return [(int(a), int(b)) for a, b in pr]


@njit(cache=True)
def _faces(Tm, N, S, Zm, k, d):
    """facets of the face with ray set S (ascending): rows of the returned mask matrix, in discovery order."""
    m, f = Tm.shape
    out = np.zeros((f, m), np.bool_); no = 0; ops = 0
    for i in range(f):
        if Zm[i]:
            continue
        ops += len(S)
        S2 = np.zeros(m, np.bool_); n2 = 0
        for q in S:
            if Tm[q, i]:
                S2[q] = True; n2 += 1
        if n2 < k - 1:
            continue
        dup = False
        for j in range(no):
            same = True
            for q in range(m):
                if out[j, q] != S2[q]:
                    same = False
                    break
            if same:
                dup = True
                break
        if dup:
            continue
        Z2 = np.ones(f, np.bool_)
        for q in range(m):
            if S2[q]:
                for ii in range(f):
                    Z2[ii] = Z2[ii] and Tm[q, ii]
        if d - _rank(N[Z2]) == k - 1:
            out[no] = S2; no += 1
        ops += d ** 3
    return out[:no], ops


def _tri_mass(c, ops):
    d = c.R.shape[1]
    if len(c.R) == d:
        estG.BUF[0].add(c.R); return 0.0
    Tm = np.ascontiguousarray(c.T); N = np.ascontiguousarray(c.N)

    def tri(S, k):
        if len(S) == k:
            return [S]
        a = S[0]
        Zm = np.all(Tm[list(S)], axis=0)
        F, o = _faces(Tm, N, np.array(S, np.int64), Zm, k, d); ops.add(int(o))
        res = []
        for row in F:
            Fs = tuple(int(q) for q in np.flatnonzero(row))
            if a in Fs:
                continue
            for simp in tri(Fs, k - 1):
                res.append((a,) + simp)
        return res

    for simp in tri(tuple(range(len(c.R))), d):
        estG.BUF[0].add(c.R[list(simp)])
    return 0.0


B._adjacent_pairs = adjacent_pairs
B.mass = _tri_mass
estG._tri_mass = _tri_mass


def _warm():
    """compile in the parent before the worker pool forks"""
    c = B.Cone.from_normals(np.eye(4))
    _adj(np.ascontiguousarray(c.T), c.N, 4)
    _faces(np.ascontiguousarray(c.T), c.N, np.arange(4, dtype=np.int64), np.zeros(4, np.bool_), 3, 4)


_warm()

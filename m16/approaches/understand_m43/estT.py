"""T85/T87: the resolve-by-impact recipe for bias-free 2nd-argmax RNNs with any T (rungs (4,3) and (4,4)).
Deductive: no random inputs. Cells are pointed polyhedral cones in R^T kept in both V- and H-representation.

Geometry (model-independent): each ordering chamber x_a > x_b > ... (class = b, the 2nd largest) times the chart
sign(x0) is split by the equidistance plane 2 x_b = x_a + x_c into the half closer to x_a = x_b and the half closer to
x_b = x_c (the two class-changing boundaries). T = 3: 24 halves (same partition as est_mech.HALVES); T = 4: 96.

Recipe (every coefficient from the weights): step-1 gates fixed by the chart; at steps 2..T-1 a gate that cuts the
cell is resolved (cell split by it) if its impact (largest pre-activation on its minority side x largest |W_hh|
entry of its column) is >= eps x the summed impact of the surely-on neurons; otherwise it is frozen at its state
on the cell's true-boundary facet. At the last step the margin rule: a cutting gate is resolved only if the logit
shift it can cause on its minority side reaches the frozen readout margin there, or the margin changes sign there with
a shift >= eps x max|margin|. Then the error = mass of the cell's wrong-class part (clip by l_true - l_o >= 0, sum
the cut-off pieces). mode='exact' resolves every cutting gate (ground truth by enumeration).

Cone masses: pulling triangulation into simplicial cones; simplicial mass = orthant probability of the dual
normals' correlation matrix: T = 3 closed form (1/8 + sum asin / 4 pi); T = 4 Plackett reduction to a 1-D integral
(1/16 + sum asin / 8 pi + sum_ij int_0^1 rho_ij asin(rho_kl.ij(t)) / (4 pi^2 sqrt(1 - t^2 rho_ij^2)) dt), evaluated
with a fixed Gauss-Legendre rule after the substitution t = 1 - (1 - s)^2 (deterministic).
"""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
from m43 import Ops  # noqa: E402

TOL = 1e-10
def _rule(n, p):
    x, w = np.polynomial.legendre.leggauss(n); s = 0.5 * (x + 1)
    return list(zip(1 - (1 - s) ** p, 0.5 * w, p * (1 - s) ** (p - 1)))      # (t, weight, dt/ds), t = 1-(1-s)^p


_RULES = {"wide": _rule(12, 2), "thin": _rule(48, 3)}      # 'thin' when max |rho| > 0.95 (peaked 1/sqrt(1-t^2 rho^2))
STATS = {"cells": 0, "cut": 0, "resolved": 0, "simplices": 0}


# ---------------------------------------------------------------------------------------------- cones
class Cone:
    __slots__ = ("R", "N")

    def __init__(self, R, N):
        self.R = R            # (m, d) unit rays
        self.N = N            # (f, d) inward normals (may be redundant)

    @staticmethod
    def from_normals(N):
        N = np.asarray(N, float)
        Rm = np.linalg.inv(N)
        R = (Rm / np.linalg.norm(Rm, axis=0)).T
        return Cone(R, N.copy())

    def tight(self):
        return np.abs(self.R @ self.N.T) < TOL * np.maximum(1, np.abs(self.N).max(1))[None, :]


def _adjacent_pairs(c, ops):
    d = c.R.shape[1]; Tm = c.tight(); m = len(c.R)
    ops.add(m * len(c.N) * (d + 1))
    out = []
    for a in range(m):
        for b in range(a + 1, m):
            Z = Tm[a] & Tm[b]; ops.add(len(c.N))
            if Z.sum() < d - 2:
                continue
            others = [q for q in range(m) if q != a and q != b]
            if any(np.all(Tm[q][Z]) for q in others):
                ops.add(len(others) * int(Z.sum())); continue
            ops.add(len(others) * int(Z.sum()))
            if np.linalg.matrix_rank(c.N[Z]) >= d - 2:
                ops.add(d ** 3); out.append((a, b))
    return out


def clip(c, n, ops):
    """(inside {n.x >= 0}, outside) cones; None for measure zero."""
    d = c.R.shape[1]
    s = c.R @ n; ops.add((d + 1) * len(c.R))
    sc = TOL * np.linalg.norm(n)
    if s.min() >= -sc:
        return c, None
    if s.max() <= sc:
        return None, c
    new = []
    for a, b in _adjacent_pairs(c, ops):
        if (s[a] > sc and s[b] < -sc) or (s[a] < -sc and s[b] > sc):
            p = abs(s[a]) * c.R[b] + abs(s[b]) * c.R[a]; p /= np.linalg.norm(p); ops.add(3 * d + 3)
            new.append(p)
    ins = [c.R[q] for q in range(len(c.R)) if s[q] >= -sc] + new
    out = [c.R[q] for q in range(len(c.R)) if s[q] <= sc] + new
    ci = Cone(np.array(ins), np.vstack([c.N, n])) if len(ins) >= d else None
    co = Cone(np.array(out), np.vstack([c.N, -n])) if len(out) >= d else None
    if ci is not None and np.linalg.matrix_rank(ci.R) < d:
        ci = None
    if co is not None and np.linalg.matrix_rank(co.R) < d:
        co = None
    ops.add(2 * d ** 3)
    return ci, co


# ------------------------------------------------------------------------------------------- masses
def orthant(Rays, ops):
    """Gaussian mass of the simplicial cone spanned by the rows of Rays (d = 3 or 4)."""
    d = len(Rays)
    B = np.linalg.inv(Rays.T)                       # rows: dual normals, cone = {B x >= 0}
    S = B @ B.T; dg = np.sqrt(np.diag(S)); C = S / np.outer(dg, dg)
    ops.add(d ** 3 + d * d * d + d * d)
    STATS["simplices"] += 1
    iu = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if d == 2:
        return 0.25 + math.asin(np.clip(C[0, 1], -1, 1)) / (2 * math.pi)
    if d == 3:
        ops.add(3 * 2)
        return 0.125 + sum(math.asin(np.clip(C[i, j], -1, 1)) for i, j in iu) / (4 * math.pi)
    assert d == 4
    p = 1 / 16 + sum(math.asin(np.clip(C[i, j], -1, 1)) for i, j in iu) / (8 * math.pi); ops.add(6 * 2)
    if abs(np.linalg.det(Rays)) < 1e-12:
        return 0.0
    rmax = max(abs(C[i, j]) for i, j in iu)
    acc = 0.0
    for t, wgt, jac in _RULES["thin" if rmax > 0.95 else "wide"]:
        Rt = np.eye(4) + t * (C - np.eye(4))
        Om = np.linalg.inv(Rt); ops.add(4 ** 3 + 16)
        for i, j in iu:
            k, l = [q for q in range(4) if q != i and q != j]
            pc = -Om[k, l] / math.sqrt(Om[k, k] * Om[l, l])
            r = C[i, j]
            acc += wgt * jac * r * math.asin(np.clip(pc, -1, 1)) / math.sqrt(max(1e-300, 1 - (t * r) ** 2))
            ops.add(12)
    return p + acc / (4 * math.pi ** 2)


def mass(c, ops):
    """Gaussian mass of a pointed polyhedral cone (T = 3 or 4) by pulling triangulation."""
    d = c.R.shape[1]
    if len(c.R) == d:
        return orthant(c.R, ops)
    Tm = c.tight(); ops.add(len(c.R) * len(c.N) * (d + 1))

    def faces_of(S, Zset, k):
        """facets (k-1 dim faces) of the k-dim face with ray index set S."""
        out = set()
        for i in range(len(c.N)):
            if i in Zset:
                continue
            S2 = tuple(q for q in S if Tm[q, i])
            if len(S2) >= k - 1 and S2 not in out:
                if np.linalg.matrix_rank(c.R[list(S2)]) == k - 1:
                    out.add(S2)
            ops.add(len(S))
        return out

    def tri(S, k):
        if len(S) == k:
            return [S]
        a = S[0]
        Z = set(i for i in range(len(c.N)) if all(Tm[q, i] for q in S))
        res = []
        for F in faces_of(S, Z, k):
            if a in F:
                continue
            for simp in tri(F, k - 1):
                res.append((a,) + simp)
        return res

    tot = 0.0
    for simp in tri(tuple(range(len(c.R))), d):
        tot += orthant(c.R[list(simp)], ops)
    return tot


# -------------------------------------------------------------------------------------- family halves
def halves(T):
    E = np.eye(T); out = []
    for sigma in (1.0, -1.0):
        for perm in itertools.permutations(range(T)):
            on = [E[perm[q]] - E[perm[q + 1]] for q in range(T - 1)]
            base = Cone.from_normals(on + [sigma * E[0]])
            a, b, c_ = perm[0], perm[1], perm[2]
            split = 2 * E[b] - E[a] - E[c_]               # >= 0: closer to x_a = x_b
            for sgn, ntrue in ((1.0, E[a] - E[b]), (-1.0, E[b] - E[c_])):
                h, _ = clip(base, sgn * split, Ops())
                if h is not None:
                    out.append(dict(sigma=sigma, perm=perm, truth=b, n_true=ntrue, cone=h))
    return out


_HALVES = {}


def _ref(c, n_true, ops):
    on = np.abs(c.R @ n_true) < 1e-9; ops.add((len(n_true) + 1) * len(c.R))
    d = c.R.shape[1]
    return c.R[on].mean(0) if on.sum() >= d - 1 else c.R.mean(0)


def _split_impact(cells, A, w, eps, ops, exact):
    out = []
    for c, ref in cells:
        S = c.R @ A.T; ops.add(len(c.R) * A.shape[0] * (A.shape[1] + 1))
        g = np.zeros(len(A)); cut = []; scale = 0.0
        for k in range(len(A)):
            sc = TOL * np.linalg.norm(A[k])
            if S[:, k].min() >= -sc:
                g[k] = 1.0; scale += S[:, k].max() * w[k]
            elif S[:, k].max() <= sc:
                g[k] = 0.0
            else:
                g[k] = float(A[k] @ ref > 0); ops.add(A.shape[1] + 1)
                depth = S[:, k].max() if g[k] == 0 else -S[:, k].min()
                cut.append((depth * w[k], k))
        STATS["cut"] += len(cut)
        chosen = [k for imp, k in sorted(cut, reverse=True) if exact or imp >= eps * scale]
        STATS["resolved"] += len(chosen)
        pieces = [(c, g)]
        for k in chosen:
            nxt = []
            for p, gg in pieces:
                ci, co = clip(p, A[k], ops)
                if ci is not None:
                    g1 = gg.copy(); g1[k] = 1; nxt.append((ci, g1))
                if co is not None:
                    g0 = gg.copy(); g0[k] = 0; nxt.append((co, g0))
            pieces = nxt
        out += pieces
    return out


def _margin_cells(c, A, Wo, truth, n_true, eps, ops, exact, done=()):
    ref = _ref(c, n_true, ops)
    S = c.R @ A.T; ops.add(len(c.R) * A.shape[0] * (A.shape[1] + 1))
    g = np.zeros(len(A)); cut = []
    for k in range(len(A)):
        sc = TOL * np.linalg.norm(A[k])
        if S[:, k].min() >= -sc:
            g[k] = 1.0 if S[:, k].max() > sc else float(A[k] @ ref > 0)
        elif S[:, k].max() <= sc:
            g[k] = 0.0
        else:
            g[k] = float(A[k] @ ref > 0); ops.add(A.shape[1] + 1)
            if k not in done:
                cut.append(k)
    if not done:
        STATS["cut"] += len(cut)
    if not cut:
        return [(c, g)]
    if exact:
        k = cut[0]
    else:
        w = np.max(np.abs(Wo[truth][None, :] - Wo), 0)
        L = Wo @ (g[:, None] * A); ops.add(Wo.shape[0] * int(g.sum()) * A.shape[1])
        pairs = _adjacent_pairs(c, ops)
        best = None
        for k in cut:
            sk = S[:, k]
            mino = (sk > 0) != (g[k] > 0)
            P = [c.R[q] for q in np.flatnonzero(mino)]; sh = [abs(sk[q]) * w[k] for q in np.flatnonzero(mino)]
            for a, b in pairs:
                if (sk[a] > 0) != (sk[b] > 0):
                    pp = abs(sk[a]) * c.R[b] + abs(sk[b]) * c.R[a]; P.append(pp / np.linalg.norm(pp)); sh.append(0.0)
                    ops.add(3 * A.shape[1] + 3)
            P = np.array(P); sh = np.array(sh)
            if g.sum() == 0:
                mm = np.zeros(len(P))
            else:
                Z = P @ L.T; mm = Z[:, truth] - np.max(np.delete(Z, truth, 1), 1); ops.add(len(P) * L.size)
            score = np.max(sh / (np.abs(mm) + 1e-300))
            flips = mm.min() < 0 < mm.max() and sh.max() >= eps * np.abs(mm).max()
            ops.add(3 * len(P))
            if score >= 1 or flips:
                if best is None or score > best[0]:
                    best = (score, k)
        if best is None:
            return [(c, g)]
        k = best[1]
    STATS["resolved"] += 1
    out = []
    for q in clip(c, A[k], ops):
        if q is not None:
            out += _margin_cells(q, A, Wo, truth, n_true, eps, ops, exact, tuple(done) + (k,))
    return out


def estT(model, ops=None, eps=0.1, mode="recipe", info=None):
    """mode 'recipe' (T85: adaptive impact at steps 2..T-1, margin rule at step T) or 'exact'."""
    Wi, Wh, Wo = model
    T = Wo.shape[0]
    ops = ops if ops is not None else Ops()
    if T not in _HALVES:
        _HALVES[T] = halves(T)
    exact = mode == "exact"
    E = np.eye(T); err = 0.0
    for h in _HALVES[T]:
        g1 = (h["sigma"] * Wi > 0).astype(float); ops.add(len(Wi))
        A1 = np.outer(Wi, E[0])
        err += _finish(h["cone"], g1, A1, 1, model, h["truth"], h["n_true"], eps, ops, exact)
    if info is not None:
        info["err"] = err
    return 1.0 - err


def _finish(c, g, A, t, model, truth, n_true, eps, ops, exact):
    """Continue the step recursion for one cell from step t (0-based input index) with previous rows A."""
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]; E = np.eye(T)
    At = Wh @ (g[:, None] * A) + np.outer(Wi, E[t]); ops.add(H * int(g.sum()) * t)
    err = 0.0
    if t < T - 1:
        for c2, g2 in _split_impact([(c, _ref(c, n_true, ops))], At, np.abs(Wh).max(0), eps, ops, exact):
            err += _finish(c2, g2, At, t + 1, model, truth, n_true, eps, ops, exact)
    else:
        for c2, g2 in _margin_cells(c, At, Wo, truth, n_true, eps, ops, exact):
            err += _cell_error(c2, g2, At, Wo, truth, ops)
    return err


def _cell_error(c, g, A, Wo, truth, ops):
    STATS["cells"] += 1
    if g.sum() == 0:
        return 0.0 if truth == 0 else mass(c, ops)
    L = Wo @ (g[:, None] * A); ops.add(Wo.shape[0] * int(g.sum()) * A.shape[1])
    err = 0.0; q = c
    for o in range(Wo.shape[0]):
        if o == truth:
            continue
        ci, co = clip(q, L[truth] - L[o], ops)
        if co is not None:
            err += mass(co, ops)
        q = ci
        if q is None:
            break
    return err

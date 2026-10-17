"""T73: fibre-traced wedge estimator for bias-free 2nd-argmax RNNs with T = 4 inputs (any H). Deductive.

Same recipe as fibre.py (T = 3) one dimension up:
  * Geometry (model-independent, exact partition of S^3): every x off the true boundaries is assigned to the nearer
    of the two class-changing boundaries of its ordering chamber (x_a > x_b > x_c > x_d has class b; the boundaries
    x_a = x_b and x_b = x_c) and written x ~ p + u n' with p on that boundary hyperplane x_i = x_j, n' = +-n,
    u = tan(phi). The boundary point p ranges over a FACE of the hyperplane: (one above) x_k > m > x_l, or (none
    above) m > x_k, x_l (split by x_k vs x_l), m = x_i = x_j. The fibre stops at the CLAMP: the equidistance plane
    with the chamber's other boundary, and (one-above faces) the order plane x_lo = x_l where the moving lower
    coordinate meets the coordinate below it. Volume element on S^3: cos^2(phi) dphi dOmega_2(omega), |S^3| = 2 pi^2,
    so a wrong fibre interval (u_lo, u_hi) contributes G(u_hi) - G(u_lo), G(u) = (atan u + u / (1 + u^2)) / 2.
  * Mechanism (from the weights): the gate pattern ON each face is enumerated exactly (2-D cells: the face polygon,
    in hyperplane coordinates, clipped by each step's gate rows restricted to the hyperplane, step by step). On the
    fibre the network is piecewise linear in u; every breakpoint (gate switch u_g = -(A.p)/(A.n'), class tie
    u_cd = -(w.p)/(w.n'), clamp) is u = alpha . omega, linear on the face. The trace at one point of a cell gives
    the wrong intervals and the order constraints (great circles (alpha_1 - alpha_2) . omega = 0) under which
    they keep their combinatorics; the cell is clipped by those constraints, and G(alpha . omega) is integrated
    over the resulting spherical polygon with a deterministic Duffy-Gauss rule (smooth integrand).
  * Tiers as in fibre.py: F0 frozen pattern, F1 traced until the end of the first correct piece, F1a attached wedge
    only, FX traced to the clamp (exact up to the cubature error; validation).
accuracy = 1 - (sum of wrong masses) / (2 pi^2).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
from m43 import clip as _clip_raw  # noqa: E402
from fiber3 import Ops  # noqa: E402
from fibre import rows_at  # noqa: E402

S3 = 2 * math.pi ** 2
EPS = 1e-11


def clip(poly, n, ops=None):
    """m43.clip with vertices within EPS of the plane snapped onto it (no sliver loops)."""
    nn_ = np.linalg.norm(n)
    s = [float(n @ v) / nn_ for v in poly]
    if ops is not None:
        ops.add(3 * len(poly))
    if min(s) >= -EPS:
        return poly, []
    if max(s) <= EPS:
        return [], poly
    ins, out = [], []
    m = len(poly)
    for i in range(m):
        j = (i + 1) % m
        vi, si, sj = poly[i], s[i], s[j]
        si = 0.0 if abs(si) < EPS else si
        sj = 0.0 if abs(sj) < EPS else sj
        if si >= 0:
            ins.append(vi)
        if si <= 0:
            out.append(vi)
        if (si > 0 and sj < 0) or (si < 0 and sj > 0):
            p = abs(si) * poly[j] + abs(sj) * vi
            p = p / np.linalg.norm(p)
            if ops is not None:
                ops.add(10)
            ins.append(p); out.append(p)
    return (ins if len(ins) >= 3 else []), (out if len(out) >= 3 else [])
NQ = 6                          # Gauss-Legendre points per Duffy direction (validated in check4.py)


def _basis(n):
    """Orthonormal basis (4 x 3) of the hyperplane n^perp."""
    M = np.linalg.svd(n[None, :])[2]
    return M[1:].T


def _lune_tris(cons, splitter):
    """Spherical polygons (in R^3 coords) of {c . w >= 0 for c in cons} (two constraints = lune) split by +-splitter."""
    out = []
    for sg in (1.0, -1.0):
        N = np.array(list(cons) + [sg * splitter])
        R = np.linalg.inv(N)
        poly = [R[:, m] / np.linalg.norm(R[:, m]) for m in range(3)]
        out.append(poly)
    return out


def families4():
    E = np.eye(4); out = []
    for i in range(4):
        for j in range(i + 1, 4):
            k, l = [q for q in range(4) if q not in (i, j)]
            n = (E[i] - E[j]) / math.sqrt(2)
            B = _basis(n)
            mid = 0.5 * (E[i] + E[j])
            faces = []
            for kk, ll in ((k, l), (l, k)):
                # one above: x_kk > m > x_ll
                faces.append(("one", kk, ll, [E[kk] - mid, mid - E[ll]]))
                # none above, x_kk > x_ll: m > x_kk > x_ll
                faces.append(("none", kk, ll, [mid - E[kk], E[kk] - E[ll]]))
            splitter = B.T @ (np.ones(4) / 2.0 - E[i] * 0.0)       # sum of coordinates (generic splitter)
            for kind, kk, ll, cons in faces:
                cb = [B.T @ c for c in cons]
                polys = _lune_tris(cb, B.T @ np.ones(4))
                sides = []
                for sg, hi, lo in ((1, i, j), (-1, j, i)):
                    nn = sg * n
                    if kind == "one":        # x_kk > x_hi > x_lo > x_ll : class hi
                        truth = hi; clamps = [E[kk] - 2 * E[hi] + E[lo], E[lo] - E[ll]]
                    else:                    # x_hi > x_lo > x_kk > x_ll : class lo
                        truth = lo; clamps = [2 * E[lo] - E[hi] - E[kk]]
                    sides.append((nn, truth, clamps))
                out.append(dict(pair=(i, j), kind=kind, kk=kk, ll=ll, B=B, n=n, polys=polys, sides=sides))
    return out


FAMS4 = families4()


# ----------------------------------------------------------------------------- per-piece state
class State:
    __slots__ = ("R", "L", "live", "gal", "ok", "Lb", "Ln")

    def __init__(self, model, x, B, nn, ops):
        R, L, live = rows_at(model, x, ops)
        self.R, self.L, self.live = R, L, live
        Rb = R @ B; Rn = R @ nn; ops.add(4 * 4 * len(R))
        ok = np.abs(Rn) > 1e-300
        with np.errstate(divide="ignore", invalid="ignore"):
            self.gal = np.where(ok[:, None], -Rb / Rn[:, None], 0.0)
        ops.add(3 * len(R))
        self.ok = ok
        self.Lb = L @ B; self.Ln = L @ nn; ops.add(4 * 4 * L.shape[0])


def _fn(w, B, nn):
    d = w @ nn
    return -(B.T @ w) / d


def trace(model, st0, B, nn, truth, cfn, w0, mode, ops, cache, look=1, info=None):
    """Fibre at face point w0 (unit, hyperplane coords). cfn: clamp fns (list of 3-vectors).
    Returns (intervals [(lo or None, hi)], constraints list of 3-vectors c with sign(c . w0) to keep)."""
    p = B @ w0
    nn_ = nn
    cv = [float(c @ w0) for c in cfn]; ic = int(np.argmin(cv)); ucl = cv[ic]; fcl = cfn[ic]
    cons = [cfn[q] - fcl for q in range(len(cfn)) if q != ic]
    st = st0; f_u = None; u = 0.0; out = []; n_ok = 0
    T = st.L.shape[0]
    for _ in range(64):
        ug = st.gal @ w0; ops.add(3 * len(ug))
        kk = -1; unext = math.inf; ahead = None
        if mode != "F0":
            if f_u is None:
                ahead = st.ok & (ug > 1e-15)
                cons.append(st.gal[st.ok])
            else:
                d = st.gal - f_u
                same = st.ok & (np.abs(d).sum(1) < 1e-12 * (1 + np.abs(f_u).sum()))
                ahead = st.ok & ~same & (ug > u)
                cons.append(d[st.ok & ~same])
            if ahead.any():
                idx = np.flatnonzero(ahead); kk = int(idx[np.argmin(ug[idx])]); unext = float(ug[kk])
        if unext >= ucl:
            kk = -1; unext = ucl; f_next = fcl
            if ahead is not None and ahead.any():
                cons.append(st.gal[ahead] - fcl)
        else:
            f_next = st.gal[kk]
            cons.append(fcl - f_next)
            if mode != "F1a":
                o = np.flatnonzero(ahead); o = o[o != kk]
                cons.append(st.gal[o] - f_next)
        a = st.Lb @ w0; b = st.Ln; ops.add(3 * T)
        edges = [(u, f_u)]
        if st.live:
            for dcl in range(T):
                if dcl == truth:
                    continue
                den = b[dcl] - b[truth]
                if abs(den) < 1e-300:
                    cons.append((st.Lb[truth] - st.Lb[dcl])[None]); continue
                fd = (st.Lb[truth] - st.Lb[dcl]) / den; ops.add(4)
                cons.append((fd if f_u is None else fd - f_u)[None]); cons.append((fd - f_next)[None])
                v = float(fd @ w0)
                if u < v < unext:
                    edges.append((v, fd))
            if len(edges) > 2:
                for x in range(1, len(edges)):
                    for y in range(x + 1, len(edges)):
                        cons.append((edges[x][1] - edges[y][1])[None])
        edges.sort(key=lambda e: e[0])
        edges.append((unext, f_next))
        if mode == "F1a":
            stop = None
            for q in range(len(edges) - 1):
                vm = 0.5 * (edges[q][0] + edges[q + 1][0])
                cls = int(np.argmax(a + vm * b)) if st.live else 0; ops.add(2 * T)
                if cls == truth:
                    stop = q; break
            if stop == 0:
                break
            if stop is not None:
                f_end = edges[stop][1]
                if kk >= 0:
                    cons.append(st.gal[ahead] - f_end)
                _append(out, f_u, f_end)
                break
            if kk >= 0:
                o = np.flatnonzero(ahead); o = o[o != kk]
                cons.append(st.gal[o] - f_next)
            _append(out, f_u, f_next)
            if kk < 0:
                break
        else:
            cls_end = truth
            for q in range(len(edges) - 1):
                vm = 0.5 * (edges[q][0] + edges[q + 1][0])
                cls = int(np.argmax(a + vm * b)) if st.live else 0; ops.add(2 * T)
                if cls != truth:
                    _append(out, edges[q][1], edges[q + 1][1])
                cls_end = cls
            piece_ok = all((int(np.argmax(a + 0.5 * (edges[q][0] + edges[q + 1][0]) * b)) == truth) if st.live
                           else truth == 0 for q in range(len(edges) - 1))
            n_ok = n_ok + 1 if piece_ok else 0
            if info is not None:
                info["f_stop"], info["u_stop"] = f_next, unext
            if kk < 0 or (mode == "F1" and cls_end == truth and (look <= 1 or n_ok >= look)):
                break
        key = (id(st), kk)
        if key not in cache:
            x = p + (unext * (1 + 1e-9) + 1e-12) * nn_
            cache[key] = State(model, x, B, nn_, ops)
        st = cache[key]; u = unext; f_u = f_next
    C = [c if c.ndim == 2 else c[None] for c in cons]
    C = np.vstack(C) if C else np.zeros((0, 3))
    return out, C


def _append(out, lo, hi):
    if out and out[-1][1] is lo:
        out[-1] = (out[-1][0], hi)
    else:
        out.append((lo, hi))


# ----------------------------------------------------------------------------- cubature of G over spherical polygons
_x, _w = np.polynomial.legendre.leggauss(NQ)
_x = 0.5 * (_x + 1); _w = 0.5 * _w
_A, _Bq = np.meshgrid(_x, _x, indexing="ij")
_WA = np.outer(_w, _w) * (1 - _A)                     # Duffy: s = a, t = (1 - a) b, Jacobian (1 - a)
_S = _A.ravel(); _T = ((1 - _A) * _Bq).ravel(); _WW = _WA.ravel()


def G(u):
    return 0.5 * (np.arctan(u) + u / (1 + u * u))


def _duffy(nq):
    x, w = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * (x + 1); w = 0.5 * w
    A, Bq = np.meshgrid(x, x, indexing="ij")
    return A.ravel(), ((1 - A) * Bq).ravel(), (np.outer(w, w) * (1 - A)).ravel()


NQ_CORR = 3
_S3, _T3, _W3 = _duffy(NQ_CORR)
_RULES = {n: _duffy(n) for n in (3, 6, 10, 16)}
ADAPT = True        # remainder order from the largest |u| at the vertices (|u| <= 0.1: 3, <= 0.5: 6, <= 2: 10, else 16)


def vec_area(poly, ops):
    """int_P omega dOmega = 1/2 sum_e theta_e n_e for a convex spherical polygon (closed form, check4.py)."""
    A = np.zeros(3)
    for q in range(len(poly)):
        a, b = poly[q], poly[(q + 1) % len(poly)]
        x = np.cross(a, b); s = math.sqrt(x @ x)
        if s < 1e-300:
            continue
        A += (0.5 * math.atan2(s, a @ b) / s) * x
    ops.add(20 * len(poly) + 3)
    return A if A @ poly[0] + A @ poly[1] > 0 else -A


def poly_int(poly, fns, signs, ops, mode="closed"):
    """sum_f sign_f * int_poly G(f . omega) dOmega. 'closed': the linear part of G exactly (vector area) plus the
    cubic-and-higher remainder G(u) - u by a 3x3 Duffy-Gauss rule per fan triangle; 'cubature': NQ x NQ rule."""
    if len(poly) < 3 or not fns:
        return 0.0
    F = np.array(fns); sg = np.array(signs)
    if mode == "closed":
        VA = vec_area(poly, ops)
        tot = float((F @ VA) @ sg); ops.add(3 * len(fns))
        S_, T_, W_ = _S3, _T3, _W3
        if ADAPT:
            umax = float(np.max(np.abs(np.array(poly) @ F.T))); ops.add(3 * len(poly) * len(fns))
            nq = 3 if umax <= 0.1 else 6 if umax <= 0.5 else 10 if umax <= 2 else 16
            S_, T_, W_ = _RULES[nq]
        g = lambda u: G(u) - u
    else:
        tot = 0.0; S_, T_, W_ = _S, _T, _WW; g = G
    v0 = poly[0]
    for q in range(1, len(poly) - 1):
        v1, v2 = poly[q], poly[q + 1]
        det = float(np.dot(v0, np.cross(v1, v2)))
        if abs(det) < 1e-18:
            continue
        W = v0[None] + S_[:, None] * (v1 - v0)[None] + T_[:, None] * (v2 - v0)[None]
        r = np.sqrt((W * W).sum(1))
        U = (W @ F.T) / r[:, None]
        tot += abs(det) * float(((g(U) @ sg) / r ** 3) @ W_)
        ops.add(len(S_) * (9 + 5 + len(fns) * (4 + 6)) + 12)
    return tot


def cells_on_face(model, poly, B, ops):
    """Exact gate-pattern cells of a face polygon: [(cell polygon, representative point in R^4)]."""
    H = len(model[0]); T = model[2].shape[0]
    cells = [poly]
    for t in range(T):
        nxt = []
        for c in cells:
            w = np.mean(c, 0); w /= np.linalg.norm(w)
            R, _, _ = rows_at(model, B @ w, ops)
            rows = R[t * H:(t + 1) * H] @ B; ops.add(4 * 3 * H)
            pieces = [c]
            for r in rows:
                nx2 = []
                for pc in pieces:
                    a_, b_ = clip(pc, r, ops)
                    nx2 += [q for q in (a_, b_) if q]
                pieces = nx2
            nxt += pieces
        cells = nxt
    return cells


def face_err(model, fam, side, mode, ops, stats):
    B = fam["B"]; nn, truth, clamps = side
    cfn = [_fn(w, B, nn) for w in clamps]; ops.add(16 * len(clamps))
    tot = 0.0
    for poly in fam["polys"]:
        for cell in cells_on_face(model, poly, B, ops):
            w = np.mean(cell, 0); w /= np.linalg.norm(w)
            cache = {}
            st0 = State(model, B @ w, B, nn, ops)
            stack = [cell]
            while stack:
                cur = stack.pop()
                w0 = np.mean(cur, 0); w0 /= np.linalg.norm(w0); ops.add(3 * len(cur) + 5)
                iv, C = trace(model, st0, B, nn, truth, cfn, w0, mode, ops, cache)
                stats["traces"] += 1
                if len(C):
                    s0 = C @ w0; ops.add(3 * len(C))
                    keep = np.abs(C).sum(1) > 1e-13
                    for c, s in zip(C[keep], s0[keep]):
                        cc = c if s > 0 else -c
                        a_, b_ = clip(cur, cc, ops)
                        if b_:
                            stack.append(b_)
                        cur = a_
                        if not cur:
                            break
                if not cur or not iv:
                    continue
                fns = []; sg = []
                for lo, hi in iv:
                    fns.append(hi); sg.append(1.0)
                    if lo is not None:
                        fns.append(lo); sg.append(-1.0)
                tot += poly_int(cur, fns, sg, ops)
                stats["pieces"] += 1
    return tot


def est_fibre4(model, mode="F1", ops=None, info=None):
    ops = ops if ops is not None else Ops()
    stats = {"traces": 0, "pieces": 0}
    tot = 0.0
    for fam in FAMS4:
        for side in fam["sides"]:
            tot += face_err(model, fam, side, mode, ops, stats)
    if info is not None:
        info.update(stats)
    return 1.0 - tot / S3

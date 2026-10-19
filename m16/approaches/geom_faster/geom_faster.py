"""T452: faster exact geometry for tier-2 cells. Drop-in for geom_fast.cell_geometry_fast, bitwise identical outputs.

Change (so far): simplex volumes by the staircase identity of the pulling triangulation instead of one 9 x 9
determinant per simplex. For a simplex (v_0, ..., v_n) of rigour's pulling triangulation, pick rows r_l (l = 1..n)
tight on v_l, ..., v_n and not on v_{l-1}. Such rows exist: the facet row of level l, or for the sorted base
simplex face any row of its face lattice. Then A_R [v_0 - v_n, ..., v_{n-1} - v_n] is lower triangular with diagonal
slack_{r_l}(v_{l-1}) > 0, so

    vol = prod_l slack_{r_l}(v_{l-1}) / (n! |det A_R|).

The slacks are exact rationals (one exact matrix product for all vertices and rows); |det A_R| is an arb ball per
distinct row set. Everything is converted to double-double with a bounded error, and the product is evaluated in
double-double (Joldes, Muller, Popescu 2017: DWTimesDW <= 7u^2, DWDivDW <= 15u^2; we charge 2^-100 = 64u^2 per
operation). The float enclosure is the pair of floats around the result's error ball when no float lies in it.
Otherwise the reference's own decision (geom_fast's arb ball for that simplex, then its exact path) is replayed, so
vlo / vhi are bitwise those of geom_fast.volumes_iv."""
import math
import os
import sys

import numpy as np
from flint import arb, arb_mat, ctx, fmpq, fmpq_mat
from numba import njit
from scipy.spatial import HalfspaceIntersection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
import geom_fast as GF  # noqa: E402
import rigour as R  # noqa: E402

STATS = dict(vol_staircase=0, vol_replayed=0, vol_delegated=0)
_REF_VOL = GF.volumes_iv                                # the reference (geom_fast is monkeypatched while in use)
EPS_OP = 2.0 ** -100


# ---------------------------------------------------------------- exact rational / arb -> double-double
def q2dd(x):
    """(hi, lo, err): |x - (hi + lo)| <= err, hi = round(x), lo = round(x - hi); err <= 2^-104 |x| (normal range)."""
    p, q = int(x.p), int(x.q)
    hi = p / q
    a, b = hi.as_integer_ratio()
    rn, rd = p * b - a * q, q * b                       # x - hi exactly
    lo = rn / rd
    return hi, lo, abs(lo) * 2.0 ** -52 + 2.0 ** -1074


def _arbq(x):
    m, e = x.man_exp(); m, e = int(m), int(e)
    return fmpq(m * 2 ** e) if e >= 0 else fmpq(m, 2 ** -e)


# ---------------------------------------------------------------- double-double kernels (Dekker / Knuth, no FMA)
@njit(cache=True, inline="always")
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@njit(cache=True, inline="always")
def _fast_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


@njit(cache=True, inline="always")
def _split(a):
    c = 134217729.0 * a
    h = c - (c - a)
    return h, a - h


@njit(cache=True, inline="always")
def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


@njit(cache=True, inline="always")
def _dd_mul(xh, xl, yh, yl):
    p, e = _two_prod(xh, yh)
    e += xh * yl + xl * yh
    return _fast_two_sum(p, e)


@njit(cache=True, inline="always")
def _dd_div(xh, xl, yh, yl):
    q1 = xh / yh
    ph, pl = _dd_mul(q1, 0.0, yh, yl)
    rh, rl = _two_sum(xh, -ph)
    rl = rl - pl + xl
    q2 = (rh + rl) / yh
    return _fast_two_sum(q1, q2)


@njit(cache=True)
def _chains(S, incm, m):
    """Per simplex the staircase row set as a bitmask (one uint64 word; m <= 63) and the rows, or -1 if none."""
    N, V = S.shape
    n = V - 1
    rows = np.empty((N, n), np.int64)
    mask = np.zeros(N, np.uint64)
    ok = np.ones(N, np.bool_)
    for t in range(N):
        acc = incm[S[t, n]]
        mk = np.uint64(0)
        for l in range(n, 0, -1):                     # rows tight on v_l..v_n, not on v_{l-1}
            acc = acc & incm[S[t, l]]
            cand = acc & ~incm[S[t, l - 1]]
            if cand == 0:
                ok[t] = False
                break
            r = 0
            while ((cand >> np.uint64(r)) & np.uint64(1)) == np.uint64(0):
                r += 1
            rows[t, l - 1] = r
            mk |= np.uint64(1) << np.uint64(r)
        mask[t] = mk
    return rows, mask, ok


@njit(cache=True)
def _vol_dd(S, rows, ok, slh, sll, sle, dh, dl, de, fact):
    """Double-double volume per simplex and its absolute error bound; brackets lo/hi (floats) or flag (need[t])."""
    N, V = S.shape
    n = V - 1
    lo = np.empty(N); hi = np.empty(N); need = np.zeros(N, np.bool_)
    eps_op = 2.0 ** -100
    for t in range(N):
        if not ok[t] or dh[t] == 0.0:
            need[t] = True; lo[t] = 0.0; hi[t] = 0.0
            continue
        ph, pl = 1.0, 0.0
        rel = 0.0                                       # relative input error
        for l in range(n):
            v = S[t, l]; r = rows[t, l]
            a, b = slh[v, r], sll[v, r]
            if not (a > 0.0):
                need[t] = True
                break
            ph, pl = _dd_mul(ph, pl, a, b)
            rel += sle[v, r] / a
        if need[t]:
            lo[t] = 0.0; hi[t] = 0.0
            continue
        qh, ql = _dd_mul(dh[t], dl[t], fact, 0.0)
        xh, xl = _dd_div(ph, pl, qh, ql)
        rel += de[t] / dh[t]
        err = abs(xh) * (1.000001 * (rel + (n + 3) * eps_op) * 2.0)        # first-order sum x 2 (covers products)
        if not (xh > 0.0) or not np.isfinite(err):
            need[t] = True; lo[t] = 0.0; hi[t] = 0.0
            continue
        # floats around x = xh + xl; certified only if no float lies within err of x
        if xl > 0.0:
            fl, fh = xh, np.nextafter(xh, np.inf)
            dlow = xl                                    # x - fl (xl exact as a float difference)
            dupp = (fh - xh) - xl                        # fh - x: fh - xh exact (one ulp)
        elif xl < 0.0:
            fl, fh = np.nextafter(xh, -np.inf), xh
            dlow = (xh - fl) + xl
            dupp = -xl
        else:
            need[t] = True; lo[t] = 0.0; hi[t] = 0.0
            continue
        if dlow > 2.0 * err and dupp > 2.0 * err:
            lo[t] = fl; hi[t] = fh
        else:
            need[t] = True; lo[t] = 0.0; hi[t] = 0.0
    return lo, hi, need


def volumes_staircase(V, inc, S, n, A, b):
    """Same output as geom_fast.volumes_iv(V, S, n) (bitwise), via the staircase identity."""
    bits = max(max(int(x.p).bit_length(), int(x.q).bit_length()) for v in V for x in v)
    m = len(A)
    if n * bits <= GF.EXACT_COST or m > 63 or len(S) == 0:      # reference's exact branch / wide row sets: delegate
        STATS["vol_delegated"] += len(S)
        return _REF_VOL(V, [tuple(x) for x in S], n)
    Sa = np.asarray(S, dtype=np.int64)
    incm = np.array([sum(1 << i for i in f) for f in inc], dtype=np.uint64)
    rows, mask, ok = _chains(Sa, incm, m)
    # exact slacks of every (vertex, row): one exact matrix product
    Aq = fmpq_mat(m, n, [x for r in A for x in r])
    Vq = fmpq_mat(n, len(V), [V[j][k] for k in range(n) for j in range(len(V))])
    SL = Aq * Vq
    needed = np.zeros((len(V), m), bool)
    needed[Sa[ok][:, :n], rows[ok]] = True
    slh = np.zeros((len(V), m)); sll = np.zeros((len(V), m)); sle = np.zeros((len(V), m))
    for v, r in zip(*np.nonzero(needed)):
        h, lw, e = q2dd(SL[r, v] - b[r])
        slh[v, r], sll[v, r], sle[v, r] = h, lw, e
    # |det A_R| per distinct row set (arb, 128 bits) -> double-double with error
    dh = np.zeros(len(S)); dl = np.zeros(len(S)); de = np.zeros(len(S))
    old = ctx.prec; ctx.prec = 128
    try:
        keys, inv = np.unique(mask[ok], return_inverse=True)
        tab = np.zeros((len(keys), 3))
        for j, key in enumerate(keys):
            rr = [i for i in range(m) if (int(key) >> i) & 1]
            ball = abs(arb_mat(n, n, [arb(x) for i in rr for x in A[i]]).det())
            if not (ball > 0):
                continue                                # 0 row stays 0: such simplices are replayed
            L, U = _arbq(ball.lower()), _arbq(ball.upper())
            h, lw, e = q2dd((L + U) / 2)
            tab[j] = (h, lw, e + float((U - L) / 2) * (1 + 2.0 ** -50) + 2.0 ** -1074)
        dh[ok], dl[ok], de[ok] = tab[inv, 0], tab[inv, 1], tab[inv, 2]
    finally:
        ctx.prec = old
    lo, hi, need = _vol_dd(Sa, rows, ok, slh, sll, sle, dh, dl, de, float(math.factorial(n)))
    idx = np.nonzero(need)[0]
    STATS["vol_staircase"] += len(S) - len(idx); STATS["vol_replayed"] += len(idx)
    if len(idx):                                       # replay the reference exactly (its arb ball, then exact)
        l2, h2 = _REF_VOL(V, [tuple(S[i]) for i in idx], n)
        lo[idx] = l2; hi[idx] = h2
    return lo, hi


# ---------------------------------------------------------------- enumeration: batched exact slacks
def enumerate_faster(Pl, v0, max_vertices=20000):
    """Same output as geom_fast.enumerate_fast (= rigour.Poly.enumerate): the qhull proposals are made exact one by
    one (exact n x n solves), but their exact slacks come from one exact matrix product; the BFS replay is
    geom_fast's (same neighbour rule, same fallbacks, same order)."""
    n, m = Pl.n, Pl.m
    ids = {}; pts = []; tight = []
    by_row = [set() for _ in range(m)]

    def add(v, s):
        if min(s) < 0:
            return None
        T = [i for i in range(m) if s[i] == 0]
        j = len(pts); ids[tuple(v)] = j; pts.append(v); tight.append((T, s))
        for i in T:
            by_row[i].add(j)
        return j

    def register(v):
        kv = tuple(v)
        if kv in ids:
            return ids[kv]
        return add(v, Pl.slack(v))
    props = []; seenp = set()
    for yf in GF._float_vertices(Pl):
        v = GF._exact_vertex(Pl, yf)
        if v is not None and tuple(v) not in seenp:
            seenp.add(tuple(v)); props.append(v)
    if props:
        SL = Pl.Aq * fmpq_mat(n, len(props), [props[j][k] for k in range(n) for j in range(len(props))])
        for j, v in enumerate(props):
            add(v, [SL[i, j] - Pl.bq[i] for i in range(m)])
    order = [register(v0)]; seen = {order[0]}; V = [v0]; inc = []; qi = 0
    while qi < len(order):
        j = order[qi]; v = V[qi]; qi += 1
        T, s = tight[j]
        inc.append(frozenset(T))
        nb = []
        if len(T) == n:
            Bi = None
            for c in range(n):
                rows = [T[r] for r in range(n) if r != c]
                cand = set.intersection(*sorted((by_row[i] for i in rows), key=len)) - {j}
                if len(cand) == 1:
                    nb.append(next(iter(cand)))
                    continue
                if Bi is None:
                    Bi = fmpq_mat(n, n, [x for i in T for x in Pl.A[i]]).inv()
                GF.STATS["ratio_fallback"] += 1
                nb.append(register(GF._ratio(Pl, v, T, s, [Bi[r, c] for r in range(n)])))
        else:
            for d in Pl.edges(v, T):
                GF.STATS["ratio_fallback"] += 1
                nb.append(register(GF._ratio(Pl, v, T, s, d)))
        for w in nb:
            if w not in seen:
                seen.add(w); order.append(w); V.append(pts[w])
                if len(V) > max_vertices:
                    raise RuntimeError("vertex budget")
    return V, inc


# ---------------------------------------------------------------- triangulation: same algorithm, array results
def pulling_triangulation_faster(V, inc, n, A=None):
    """Same output as geom_fast.pulling_triangulation_fast, as an int array (one row per simplex, same order).
    Faces are the same frozensets (built by intersection with per-row vertex sets), inserted into the same sets in
    the same order, so every set iteration order is the reference's."""
    P = GF.PRIME
    Vp = []
    for v in V:
        r = [GF._modp(x) for x in v]
        Vp.append(None if any(x is None for x in r) else np.array(r, dtype=object))
    Ap = None
    if A is not None:
        Ap = []
        for row in A:
            r = [GF._modp(x) for x in row]
            Ap.append(None if any(x is None for x in r) else r)
    rowset = {}
    for i, T in enumerate(inc):
        for r in T:
            rowset.setdefault(r, set()).add(i)
    rowset = {r: frozenset(x) for r, x in rowset.items()}
    exact_cache = {}

    def dim_exact(F):
        if F not in exact_cache:
            GF.STATS["rank_exact"] += 1
            exact_cache[F] = R.affine_rank([V[i] for i in sorted(F)])
        return exact_cache[F]

    def rank_p_vertices(F):
        idx = sorted(F)
        if len(idx) <= 1:
            return 0
        if any(Vp[i] is None for i in idx):
            return -1
        z = Vp[idx[0]]
        M = GF.nmod_mat(len(idx) - 1, n, [int(x) for i in idx[1:] for x in (Vp[i] - z) % P], P)
        return M.rank()

    def is_dim(G, target):
        if rank_p_vertices(G) == target:
            return True
        if Ap is not None:
            T = frozenset.intersection(*(inc[i] for i in G))
            if T and all(Ap[i] is not None for i in T):
                M = GF.nmod_mat(len(T), n, [x for i in sorted(T) for x in Ap[i]], P)
                if n - M.rank() < target:
                    return False
        return dim_exact(G) == target

    memo = {}

    def tri(F, d):
        key = (F, d)
        if key in memo:
            return memo[key]
        if len(F) == d + 1:
            res = np.array([sorted(F)], dtype=np.int64)
        else:
            v0 = min(F)
            rows = set().union(*(inc[i] for i in F))
            facets = set()
            for r in rows:
                G = F & rowset[r]
                if len(G) >= d and v0 not in G and G != F and G not in facets and is_dim(G, d - 1):
                    facets.add(G)
            parts = []
            for G in facets:
                sub = tri(G, d - 1)
                parts.append(np.column_stack([np.full(len(sub), v0, dtype=np.int64), sub]))
            res = np.vstack(parts) if parts else np.zeros((0, d + 1), np.int64)
        memo[key] = res
        return res

    F = frozenset(range(len(V)))
    d = n if rank_p_vertices(F) == n else dim_exact(F)
    if d < n:
        return np.zeros((0, n + 1), np.int64), d
    return tri(F, n), d


# ---------------------------------------------------------------- dropped-row check (rigorous float prefilter)
def check_dropped(Af, bf, drop, V, n):
    """Raises exactly when geom_fast's exact check raises (some dropped row with a.v < b at some vertex). A float
    interval enclosure of a.v - b certifies most entries; the rest are checked exactly."""
    if not drop:
        return
    Vl = np.array([[GF.q2iv(x)[0] for x in v] for v in V]).T; Vh = np.array([[GF.q2iv(x)[1] for x in v] for v in V]).T
    Al = np.array([[GF.q2iv(x)[0] for x in Af[i]] for i in drop]); Ah = np.array([[GF.q2iv(x)[1] for x in Af[i]] for i in drop])
    bh = np.array([GF.q2iv(bf[i])[1] for i in drop])
    Vm = 0.5 * Vl + 0.5 * Vh; Vr = np.maximum(Vh - Vm, Vm - Vl)
    Am = 0.5 * Al + 0.5 * Ah; Ar = np.maximum(Ah - Am, Am - Al)
    dot = Am @ Vm
    g = (n + 2) * 2.0 ** -53 / (1 - (n + 2) * 2.0 ** -53)
    err = 2.0 * (np.abs(Am) @ Vr + Ar @ np.abs(Vm) + Ar @ Vr + g * (np.abs(Am) @ np.abs(Vm) + np.abs(Am) @ Vr + Ar @ np.abs(Vm) + Ar @ Vr)) \
        * (1 + 2.0 ** -40) + 2.0 ** -1000
    ok = (dot - err) - np.abs(dot - err) * 2.0 ** -50 > bh[:, None] + np.abs(bh[:, None]) * 2.0 ** -50
    for kk, j in zip(*np.nonzero(~ok)):                 # uncertain entries: the reference's exact test
        i = drop[kk]
        if sum(Af[i][c] * V[j][c] for c in range(n)) < bf[i]:
            raise RuntimeError("dropped row violated")


# ---------------------------------------------------------------- irredundant rows: screened, LP only when close
IRR_STATS = dict(rows=0, lp=0, fallback_all=0)


def irredundant_fast(A, b, tol=1e-9, margin=1e-5):
    """Same keep list as rigour.irredundant (one HiGHS LP per row: keep row i iff the LP over the other rows and
    the box |y| <= NORM + 2 is not solved, or its value is < b_i + tol max(1, |b_i|)). One qhull run gives the float
    vertices of Q = all rows and the box. A row whose decision is clear by more than `margin` (relative to max(1,
    |b_i|, |a_i| (NORM + 2))) is decided from them: strictly redundant rows (Q_{-i} = Q, so the LP value is min over
    Q's vertices), and facet rows with a certified protrusion of Q_{-i} beyond the facet. Every other row gets the
    reference's own LP call. Soundness never depends on this list (dropped rows are re-checked exactly); identity
    with the reference is what the T452 gate tests."""
    from scipy.optimize import linprog
    Af = np.array([[float(v) for v in r] for r in A]); bf = np.array([float(v) for v in b])
    m, n = Af.shape
    BOX = R.NORM + 2
    IRR_STATS["rows"] += m

    def lp(i):
        IRR_STATS["lp"] += 1
        o = np.arange(m) != i
        r = linprog(Af[i], A_ub=-Af[o], b_ub=-bf[o], bounds=[(-BOX, BOX)] * n, method="highs")
        return r.status != 0 or r.fun < bf[i] + tol * max(1, abs(bf[i]))

    def all_lp():
        IRR_STATS["fallback_all"] += 1
        return [i for i in range(m) if lp(i)]
    Ab = np.r_[Af, np.eye(n), -np.eye(n)]; bb = np.r_[bf, -BOX * np.ones(n), -BOX * np.ones(n)]
    nrm = np.linalg.norm(Ab, axis=1)
    r = linprog(np.r_[np.zeros(n), -1.0], A_ub=np.c_[-Ab, nrm], b_ub=-bb, bounds=[(None, None)] * n + [(0, None)], method="highs")
    if r.status != 0 or r.x[-1] <= 1e-7:
        return all_lp()
    try:
        X = HalfspaceIntersection(np.c_[-Ab, bb], r.x[:-1]).intersections
    except Exception:  # noqa: BLE001
        return all_lp()
    SL = Af @ X.T - bf[:, None]                        # (m, vertices) float slacks
    SLb = Ab @ X.T - bb[:, None]
    keep = []
    for i in range(m):
        sc = max(1.0, abs(bf[i]), np.abs(Af[i]).sum() * BOX)
        thr = bf[i] + tol * max(1, abs(bf[i]))
        mn = SL[i].min()
        if mn > 1e-9 * sc:                             # not tight on Q: strictly redundant, LP value = min over Q
            if mn + bf[i] - thr > margin * sc:
                continue                               # clearly dropped
            keep.append(i) if lp(i) else None
            continue
        tv = np.nonzero(np.abs(SL[i]) <= 1e-9 * sc)[0]
        facet = False
        if len(tv) >= n:
            P = X[tv] - X[tv].mean(0)
            sv = np.linalg.svd(P, compute_uv=False)
            facet = len(sv) >= n - 1 and sv[n - 2] > 1e-6 * max(1.0, sv[0])
        if facet:                                      # protrusion of Q_{-i} beyond the facet, from its centroid
            c = X[tv].mean(0)
            u = -Af[i] / np.linalg.norm(Af[i])
            rate = Ab @ u; slc = Ab @ c - bb
            mask = np.ones(len(Ab), bool); mask[i] = False
            neg = mask & (rate < 0)
            t = np.min(slc[neg] / -rate[neg]) if neg.any() else BOX
            prot = t * np.linalg.norm(Af[i])           # decrease of a_i.y along the step
            if np.all(slc[mask] > 1e-9 * sc) and prot > margin * sc:
                keep.append(i)                         # LP value <= b_i - prot < threshold, clearly kept
                continue
        keep.append(i) if lp(i) else None
    return keep


# ---------------------------------------------------------------- drop-in for geom_fast.cell_geometry_fast
def cell_rows(tri, gates, k, norm=R.NORM, event=False):
    """The reference's slice row system before the irredundancy step: (A, b, n, yidx, I) or None if empty."""
    D = R.D
    p1, p2, p3 = tri
    I = sorted(set(range(k)) | set(tri)); d = len(I)
    g = np.array(gates).astype(int); g[k:] = -1
    if event and k != D:
        raise ValueError("event rows need k = 10")
    rows, _ = R.exact_rows(tri, g, pick=event, order=False)
    zero, one = fmpq(0), fmpq(1)
    e = lambda i: [one if c == i else zero for c in range(D)]  # noqa: E731
    rows = rows + [[a - b for a, b in zip(e(p1), e(p2))]] + [[a - b for a, b in zip(e(p3), e(l))] for l in I if l not in tri]
    yidx = [j for j in I if j != p2]; ip3 = yidx.index(p3); n = d - 1
    out = {}

    def add(c, b):
        mx = max(abs(v) for v in c)
        if mx == 0:
            return b <= 0
        c = tuple(v / mx for v in c); b = b / mx
        if c not in out or b > out[c]:
            out[c] = b
        return True
    for a in rows:
        c = [a[j] for j in yidx]; c[ip3] += a[p2]
        if not add(c, -a[p2]):
            return None
    for i in range(n):
        ee = [zero] * n; ee[i] = one
        add(ee, fmpq(-norm) - (1 if i == ip3 else 0)); add([-v for v in ee], fmpq(-norm) + (1 if i == ip3 else 0))
    return [list(c) for c in out], [out[c] for c in out], n, yidx, I


def cell_geometry_faster(tri, gates, k, norm=R.NORM, event=False, return_exact=False):
    """Same output as geom_fast.cell_geometry_fast (bitwise; S as the same int array)."""
    p1, p2, p3 = tri
    rs = cell_rows(tri, gates, k, norm, event)
    if rs is None:
        return None
    A, b, n, yidx, I = rs
    d = len(I); mlow = R.D - d
    keep = irredundant_fast(A, b)
    Af, bf = A, b
    A = [A[i] for i in keep]; b = [b[i] for i in keep]
    Pl = R.Poly(A, b); v0 = Pl.start()
    if v0 is None or v0 == "infeasible":
        raise RuntimeError("no start vertex")
    V, inc = enumerate_faster(Pl, v0)
    drop = [i for i in range(len(Af)) if i not in set(keep)]
    check_dropped(Af, bf, drop, V, n)
    S, dim = pulling_triangulation_faster(V, inc, n, A)
    if dim < n:
        return None

    def mvec(y):
        x = {j: y[q_] for q_, j in enumerate(yidx)}; x[p2] = 1 + x[p3]
        return [x[j] for j in I]
    M = [mvec(v) for v in V]
    iv = [[GF.q2iv_ref(x) for x in r] for r in M]
    Mlo = np.array([[x[0] for x in r] for r in iv]); Mhi = np.array([[x[1] for x in r] for r in iv])
    vlo, vhi = volumes_staircase(V, inc, S, n, A, b)
    res = (d, mlow, I.index(p3), Mlo, Mhi, S, vlo, vhi, len(V))
    return (res, (V, inc, S, M)) if return_exact else res

"""T415: fast exact geometry for M5 / M5t cells and event regions (drop-in for the geometry of rigour.mass_cell and
m5t_fast.cell_geometry). Outputs are identical to the reference (T415_README.md):

  enumerate_fast(Pl, v0)                 == Pl.enumerate(v0): same exact vertex list, same order, same incidences
  pulling_triangulation_fast(V, inc, n, A)  == rigour.pulling_triangulation(V, inc, n)
  volumes_iv(V, S, n)                    == [q2iv(exact fmpq simplex volume)] (outward float enclosures)
  volumes_arb(V, S, n)                   rigorous arb balls of the exact volumes (for rigour.mass_cell)
  cell_geometry_fast(tri, gates, k, ...) == m5t_fast.cell_geometry(...)

How (every decision exact; floats only propose):
- vertices: qhull (scipy HalfspaceIntersection) proposes float vertices; each is made exact by an exact n x n
  solve on rows chosen from its float slacks and kept only if exactly feasible (exact slacks of all rows give its
  exact tight set). The reference BFS (Poly.enumerate) is then replayed: at a simple vertex the neighbour across
  edge c (rows T minus T[c] tight) is the unique other vertex whose tight set contains T minus T[c] (the edge line
  meets Q in a segment whose only vertices are its two ends); if no proposal has it, the exact ratio test of the
  reference finds it. Degenerate vertices use the reference Poly.edges + ratio test. So the result is the
  reference's, by construction, whatever qhull returns.
- triangulation: the reference algorithm with the face-dimension test dim(G) == d - 1 decided by ranks modulo the
  prime p = 2^61 - 1: the affine rank of G's vertices mod p is <= the true affine rank, and dim(G) <= d - 1 always
  holds for the candidate G (a proper face of the d-face F cut by a row that is not tight on all of F), so
  rank_p == d - 1 proves dim(G) == d - 1; n - rank_p(A_{T_G}) < d - 1 (rows tight on G) proves dim(G) < d - 1;
  otherwise the exact rank decides.
- volumes: det of the 9 x 9 difference matrix in arb at 128 bits (a rigorous ball); the float enclosure is the
  pair of floats around the ball if no float lies inside it, else the exact fmpq determinant decides."""
import math
import os
import sys

import numpy as np
from flint import arb, arb_mat, ctx, fmpq, fmpq_mat, nmod_mat
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rigour as R  # noqa: E402

PRIME = 2 ** 61 - 1
STATS = dict(ratio_fallback=0, qhull_fail=0, rank_exact=0, vol_exact=0)


# ---------------------------------------------------------------- exact rational -> float enclosure
def q2iv(x):
    """Floats lo <= x <= hi, adjacent (or equal if x is a float); same as m5t_fast.q2iv. int/int true division in
    CPython is correctly rounded, so x lies between f and one neighbour; the side is decided exactly."""
    p, q = int(x.p), int(x.q)
    try:
        f = p / q
    except OverflowError:
        from m5t_fast import q2iv as slow
        return slow(x)
    a, b = f.as_integer_ratio()
    c = a * q - p * b                                    # sign of f - x (b, q > 0)
    if c == 0:
        return f, f
    if c < 0:
        return f, float(np.nextafter(f, np.inf))
    return float(np.nextafter(f, -np.inf)), f


def q2iv_ref(x):
    """Exactly m5t_fast.q2iv(x) (whose start point float(p)/float(q) may be off by an ulp, giving a 2-ulp enclosure),
    computed fast: start f0 as there, then lo = f0 if f0 <= x else the largest float <= x, hi likewise."""
    p, q = int(x.p), int(x.q)
    f0 = float(p) / float(q) if abs(p) < 2 ** 1000 and q < 2 ** 1000 else float(x)
    l, h = q2iv(x)
    a, b = f0.as_integer_ratio()
    c = a * q - p * b                                    # sign of f0 - x
    return (f0 if c <= 0 else l), (f0 if c >= 0 else h)


def _dyadic(x):
    m, e = x.man_exp(); m, e = int(m), int(e)
    return fmpq(m * 2 ** e) if e >= 0 else fmpq(m, 2 ** -e)


# ---------------------------------------------------------------- vertex enumeration
def _float_vertices(Pl):
    """Float vertex proposals from qhull; [] if qhull fails (then the exact fallback does everything)."""
    A, b = Pl.Af, Pl.bf
    nrm = np.linalg.norm(A, axis=1)
    r = linprog(np.r_[np.zeros(Pl.n), -1.0], A_ub=np.c_[-A, nrm], b_ub=-b, bounds=[(None, None)] * Pl.n + [(0, None)],
                method="highs")
    if r.status != 0 or r.x[-1] <= 1e-9:
        STATS["qhull_fail"] += 1
        return []
    try:
        hs = HalfspaceIntersection(np.c_[-A, b], r.x[:-1])
        return list(hs.intersections)
    except Exception:  # noqa: BLE001  (qhull errors: fall back to exact pivoting)
        STATS["qhull_fail"] += 1
        return []


def _exact_vertex(Pl, yf):
    """Exact vertex near the float point yf: n rows of smallest float slack, independent in float, exact solve; None
    unless exactly feasible."""
    n = Pl.n
    sl = Pl.Af @ yf - Pl.bf
    order = np.argsort(np.abs(sl) / np.maximum(1, np.abs(Pl.bf)))
    sel = [int(i) for i in order[:n]]                    # simple vertex: the n smallest slacks (exact rank checked)
    B = fmpq_mat(n, n, [v for i in sel for v in Pl.A[i]])
    if B.rank() == n:
        v = B.solve(fmpq_mat(n, 1, [Pl.bq[i] for i in sel]))
        return [v[i, 0] for i in range(n)]
    sel = []
    for i in order[:3 * n + 10]:
        if np.linalg.matrix_rank(Pl.Af[sel + [int(i)]], tol=1e-10) == len(sel) + 1:
            sel.append(int(i))
        if len(sel) == n:
            break
    if len(sel) < n:
        return None
    B = fmpq_mat(n, n, [v for i in sel for v in Pl.A[i]])
    if B.rank() < n:
        return None
    v = B.solve(fmpq_mat(n, 1, [Pl.bq[i] for i in sel]))
    return [v[i, 0] for i in range(n)]


def enumerate_fast(Pl, v0, max_vertices=20000):
    """Same output as Pl.enumerate(v0) (rigour.Poly): exact vertices in BFS order and their tight sets."""
    n, m = Pl.n, Pl.m
    ids = {}                                              # exact vertex key -> id
    pts = []; tight = []                                  # per id: point, (sorted tight rows, slacks)
    by_row = [set() for _ in range(m)]

    def register(v, kv=None):
        kv = tuple(v) if kv is None else kv
        if kv in ids:
            return ids[kv]
        s = Pl.slack(v)
        if min(s) < 0:
            return None
        T = [i for i in range(m) if s[i] == 0]
        j = len(pts); ids[kv] = j; pts.append(v); tight.append((T, s))
        for i in T:
            by_row[i].add(j)
        return j
    for yf in _float_vertices(Pl):
        v = _exact_vertex(Pl, yf)
        if v is not None:
            register(v)
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
                if Bi is None:                               # exact fallback: reference ratio test along column c
                    Bi = fmpq_mat(n, n, [x for i in T for x in Pl.A[i]]).inv()
                STATS["ratio_fallback"] += 1
                nb.append(register(_ratio(Pl, v, T, s, [Bi[r, c] for r in range(n)])))
        else:
            for d in Pl.edges(v, T):
                STATS["ratio_fallback"] += 1
                nb.append(register(_ratio(Pl, v, T, s, d)))
        for w in nb:
            if w not in seen:
                seen.add(w); order.append(w); V.append(pts[w])
                if len(V) > max_vertices:
                    raise RuntimeError("vertex budget")
    return V, inc


def _ratio(Pl, v, T, s, d):
    """The reference ratio test (rigour.Poly.enumerate): end of the edge from v in direction d."""
    Ad = Pl.Aq * fmpq_mat(Pl.n, 1, d)
    lam = None
    for i in range(Pl.m):
        a = Ad[i, 0]
        if a < 0 and i not in T:
            t = s[i] / (-a)
            if lam is None or t < lam:
                lam = t
    if lam is None:
        raise ValueError("unbounded edge (Q must be bounded by N)")
    return [v[k] + lam * d[k] for k in range(Pl.n)]


# ---------------------------------------------------------------- triangulation
def _modp(x):
    q = int(x.q) % PRIME
    if q == 0:
        return None
    return int(x.p) % PRIME * pow(q, PRIME - 2, PRIME) % PRIME


def pulling_triangulation_fast(V, inc, n, A=None):
    """rigour.pulling_triangulation with the dimension tests decided modulo PRIME (exact fallback)."""
    Vp = []
    for v in V:
        r = [_modp(x) for x in v]
        Vp.append(None if any(x is None for x in r) else r)
    Ap = None
    if A is not None:
        Ap = []
        for row in A:
            r = [_modp(x) for x in row]
            Ap.append(None if any(x is None for x in r) else r)
    exact_cache = {}

    def dim_exact(F):
        if F not in exact_cache:
            STATS["rank_exact"] += 1
            exact_cache[F] = R.affine_rank([V[i] for i in sorted(F)])
        return exact_cache[F]

    def rank_p_vertices(F):
        idx = sorted(F)
        if len(idx) <= 1:
            return 0
        if any(Vp[i] is None for i in idx):
            return -1
        z = Vp[idx[0]]
        M = nmod_mat(len(idx) - 1, n, [(Vp[i][k] - z[k]) % PRIME for i in idx[1:] for k in range(n)], PRIME)
        return M.rank()

    def is_dim(G, target):
        """dim(G) == target, for G a proper face of a (target+1)-face (so dim(G) <= target)."""
        rp = rank_p_vertices(G)
        if rp == target:
            return True
        if Ap is not None:
            T = frozenset.intersection(*(inc[i] for i in G))
            if T and all(Ap[i] is not None for i in T):
                M = nmod_mat(len(T), n, [x for i in sorted(T) for x in Ap[i]], PRIME)
                if n - M.rank() < target:
                    return False
        return dim_exact(G) == target

    memo = {}

    def tri(F, d):
        if (F, d) in memo:
            return memo[(F, d)]
        if len(F) == d + 1:
            res = [tuple(sorted(F))]
        else:
            v0 = min(F)
            rows = set().union(*(inc[i] for i in F))
            facets = set()
            for r in rows:
                G = frozenset(i for i in F if r in inc[i])
                if len(G) >= d and v0 not in G and G != F and G not in facets and is_dim(G, d - 1):
                    facets.add(G)
            res = []
            for G in facets:
                for s in tri(G, d - 1):
                    res.append((v0,) + s)
        memo[(F, d)] = res
        return res

    F = frozenset(range(len(V)))
    d = n if rank_p_vertices(F) == n else dim_exact(F)
    if d < n:
        return [], d
    return tri(F, n), d


# ---------------------------------------------------------------- volumes
def _det_balls(V, S, n, prec=128):
    old = ctx.prec; ctx.prec = prec
    try:
        VA = [[arb(x) for x in v] for v in V]
        diff = {}                                         # (apex, vertex) -> difference vector (apexes repeat)
        out = []
        for s in S:
            a = s[0]; rows = []
            for i in s[1:]:
                r = diff.get((a, i))
                if r is None:
                    r = [VA[i][k] - VA[a][k] for k in range(n)]; diff[(a, i)] = r
                rows += r
            out.append(abs(arb_mat(n, n, rows).det()))
        return out
    finally:
        ctx.prec = old


def _exact_vol(V, s, n):
    return abs(fmpq_mat(n, n, [V[s[i]][k] - V[s[0]][k] for i in range(1, n + 1) for k in range(n)]).det()) / math.factorial(n)


EXACT_COST = 3000                                        # n x vertex bits at or below this: exact volumes are cheap


def volumes_iv(V, S, n):
    """Float enclosures of the exact simplex volumes, equal to the reference's m5t_fast.q2iv(exact volume).
    Cheap cases (n x vertex bits <= EXACT_COST): the exact fmpq volume and q2iv_ref. Otherwise from a 128-bit arb ball:
    the floats around it when no float lies in it, else exactly. (For such vertices the volume's numerator or
    denominator exceeds 2^1000, where the reference's start point float(x) is the correctly rounded value, so
    both give the tight enclosure; checked on every test simplex.)"""
    bits = max(max(int(x.p).bit_length(), int(x.q).bit_length()) for v in V for x in v)
    if n * bits <= EXACT_COST:
        lo = np.empty(len(S)); hi = np.empty(len(S))
        for j, s in enumerate(S):
            lo[j], hi[j] = q2iv_ref(_exact_vol(V, s, n))
        return lo, hi
    fact = math.factorial(n)
    lo = np.empty(len(S)); hi = np.empty(len(S))
    old = ctx.prec; ctx.prec = 128
    try:
        for j, b in enumerate(_det_balls(V, S, n)):
            b = b / fact
            ok = b.is_finite() and b > 0
            if ok:
                L, U = _dyadic(b.lower()), _dyadic(b.upper())
                lL, hL = q2iv(L); lU, hU = q2iv(U)
                # no float in [L, U] <=> neither end is a float and both lie in the same float gap
                ok = lL != hL and (lL, hL) == (lU, hU)
            if ok:
                lo[j], hi[j] = lL, hU
            else:
                STATS["vol_exact"] += 1
                lo[j], hi[j] = q2iv_ref(_exact_vol(V, S[j], n))
    finally:
        ctx.prec = old
    return lo, hi


def volumes_arb(V, S, n, prec=128):
    """Rigorous arb balls containing the exact simplex volumes (usable where rigour.mass_cell uses fmpq volumes)."""
    fact = math.factorial(n)
    old = ctx.prec; ctx.prec = prec
    try:
        return [b / fact for b in _det_balls(V, S, n, prec)]
    finally:
        ctx.prec = old


# ---------------------------------------------------------------- cell geometry (drop-in for m5t_fast.cell_geometry)
def cell_geometry_fast(tri, gates, k, norm=R.NORM, event=False, return_exact=False):
    """Same output as m5t_fast.cell_geometry: (d, mlow, jp3, Mlo, Mhi, S, vlo, vhi, nV) or None. With return_exact
    also (V, inc, S, M) exactly, for the identity tests."""
    D = R.D
    p1, p2, p3 = tri
    I = sorted(set(range(k)) | set(tri)); d = len(I); mlow = D - d
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
    A = [list(c) for c in out]; b = [out[c] for c in out]
    keep = R.irredundant(A, b)
    Af, bf = A, b
    A = [A[i] for i in keep]; b = [b[i] for i in keep]
    Pl = R.Poly(A, b); v0 = Pl.start()
    if v0 is None or v0 == "infeasible":
        raise RuntimeError("no start vertex")
    V, inc = enumerate_fast(Pl, v0)
    drop = [i for i in range(len(Af)) if i not in set(keep)]
    if drop:                                              # every dropped row exactly, all vertices in one product
        Ad = fmpq_mat(len(drop), n, [v for i in drop for v in Af[i]])
        Sl = Ad * fmpq_mat(n, len(V), [V[j][kk] for kk in range(n) for j in range(len(V))])
        for kk, i in enumerate(drop):
            for j in range(len(V)):
                if Sl[kk, j] < bf[i]:
                    raise RuntimeError("dropped row violated")
    S, dim = pulling_triangulation_fast(V, inc, n, A)
    if dim < n:
        return None

    def mvec(y):
        x = {j: y[q_] for q_, j in enumerate(yidx)}; x[p2] = 1 + x[p3]
        return [x[j] for j in I]
    M = [mvec(v) for v in V]
    iv = [[q2iv_ref(x) for x in r] for r in M]
    Mlo = np.array([[x[0] for x in r] for r in iv]); Mhi = np.array([[x[1] for x in r] for r in iv])
    vlo, vhi = volumes_iv(V, S, n)
    res = (d, mlow, I.index(p3), Mlo, Mhi, np.array([list(s) for s in S]), vlo, vhi, len(V))
    return (res, (V, inc, S, M)) if return_exact else res


# ---------------------------------------------------------------- drop-in for rigour.mass_cell (no edit of rigour.py)
def patch_rigour():
    """Monkeypatch rigour.Poly.enumerate and rigour.pulling_triangulation with the fast versions (identical outputs).
    rigour.mass_cell's own volume line (exact fmpq det per simplex, ~90% of its geometry time) is unchanged; the
    one-line change proposed to T400 is to use volumes_arb(V, S, n) there (arb balls; the arb pipeline accepts them)."""
    last = {}
    orig_enum = R.Poly.enumerate

    def enum(self, v0, max_vertices=20000):
        last["A"] = self.A
        return enumerate_fast(self, v0, max_vertices)

    def tri(V, inc, n):
        return pulling_triangulation_fast(V, inc, n, last.get("A"))
    R.Poly.enumerate = enum
    R.pulling_triangulation = tri
    return orig_enum

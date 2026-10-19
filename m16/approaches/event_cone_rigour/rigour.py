"""T400: rigorous tools for the certified rare-event bound (README section "Pre-registration").

Interface
  exact_rows(tri, gates)            exact (fmpq) homogeneous rows a with a.x >= 0 for a region (tri, gates)
  mass_bounds(tri, gates, t0=1.5)   rigorous [lb, ub] on gamma(region ∩ E(t0) ∩ N), method M1 (exact slice
                                    triangulation); ub is the certified quantity
  mass_box(tri, gates, t0=1.5)      rigorous but loose ub, method M0 (bounding box of the slice)
  norm_loss(t0=1.5)                 rigorous bound on P(E(t0) \\ N), part (c)
Region = (tri, gates): gates int (10, 16) with 1 on / 0 off / -1 free. N = {g23 >= 0.1 ||x||_inf}.
All numbers that enter a bound are exact rationals (python-flint fmpq) or arb balls; floats only propose
(LP vertices, projection points) and every proposal is re-checked exactly."""
import heapq
import itertools
import math
import os
import sys

import numpy as np
from flint import arb, fmpq, fmpq_mat
from scipy.optimize import linprog, nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common  # noqa: E402

D, H = 10, 16
NORM = 10                       # ||x||_inf <= NORM * g23  (T390: x in [-1,1]^10, g23 >= 0.1)


def q(v):
    return fmpq(*float(v).as_integer_ratio())


def qmat(A):
    A = np.atleast_2d(A)
    return fmpq_mat(A.shape[0], A.shape[1], [q(v) for v in A.ravel()])


WI, WH, WO = qmat(common.Wi[:, None]), qmat(common.Wh), qmat(common.Wo)


def rows_of(M):
    return [list(r) for r in M.tolist()]


def exact_rows(tri, gates, pick=True, order=True):
    """Exact rows a (lists of fmpq, length D) with a.x >= 0. Gate rows for every assigned gate up to the first
    step with a free gate (later steps are undefined); the pick-p3 rows if all 10 steps are assigned; the
    ordering rows x_p1 >= x_p2 >= x_p3 >= lows. Exact h_k = A x is returned too (None if the prefix is partial)."""
    p1, p2, p3 = tri
    rows = []
    A = fmpq_mat(H, D)
    full = True
    for t in range(D):
        Z = WH * A
        for j in range(H):
            Z[j, t] += WI[j, 0]
        g = gates[t]
        for j in range(H):
            if g[j] == 1:
                rows.append([Z[j, c] for c in range(D)])
            elif g[j] == 0:
                rows.append([-Z[j, c] for c in range(D)])
        if (g < 0).any():
            full = False
            break
        for j in range(H):
            if g[j] == 0:
                for c in range(D):
                    Z[j, c] = 0
        A = Z
    if pick and full:
        L = WO * A
        for j in range(D):
            if j != p3:
                rows.append([L[p3, c] - L[j, c] for c in range(D)])
    if order:
        e = lambda i: [fmpq(1) if c == i else fmpq(0) for c in range(D)]  # noqa: E731
        rows.append([a - b for a, b in zip(e(p1), e(p2))])
        rows.append([a - b for a, b in zip(e(p2), e(p3))])
        for l in range(D):
            if l not in tri:
                rows.append([a - b for a, b in zip(e(p3), e(l))])
    return rows, (A if full else None)


# ---------------------------------------------------------------- slice g23 = 1, y = x without x_p2
def slice_polytope(tri, rows, NORM=NORM):
    """Q = {y in R^9: A y >= b} with x = m(y): x_j = y_j (j != p2), x_p2 = 1 + y_p3; plus N: |m_j| <= NORM.
    Rows are scaled to max |coef| = 1 and duplicates merged (keeping the tightest), all exactly.
    Returns (A rows, b, idx) or None if some row is 0 >= b > 0 (empty)."""
    p1, p2, p3 = tri
    idx = [j for j in range(D) if j != p2]
    ip3 = idx.index(p3)
    out = {}

    def add(c, b):
        m = max(abs(v) for v in c)
        if m == 0:
            return b <= 0
        c = tuple(v / m for v in c); b = b / m
        if c not in out or b > out[c]:
            out[c] = b
        return True

    for a in rows:
        c = [a[j] for j in idx]; c[ip3] += a[p2]
        if not add(c, -a[p2]):
            return None
    one, zero = fmpq(1), fmpq(0)
    for i in range(9):
        e = [zero] * 9; e[i] = one
        add(e, fmpq(-NORM) - (1 if i == ip3 else 0))                # m_j >= -NORM
        add([-v for v in e], fmpq(-NORM) + (1 if i == ip3 else 0))  # m_j <= NORM
    cs = list(out)
    return [list(c) for c in cs], [out[c] for c in cs], idx


def m_of(tri, idx, y):
    """Slice point m(y) in R^10 (exact)."""
    p2, p3 = tri[1], tri[2]
    x = [fmpq(0)] * D
    for k, j in enumerate(idx):
        x[j] = y[k]
    x[p2] = 1 + x[p3]
    return x


# ---------------------------------------------------------------- exact vertex enumeration (BFS on the graph)
class Poly:
    def __init__(self, A, b):
        self.A, self.b = A, b
        self.m, self.n = len(A), len(A[0])
        self.Aq = fmpq_mat(self.m, self.n, [v for r in A for v in r])
        self.bq = b
        self.Af = np.array([[float(v) for v in r] for r in A]); self.bf = np.array([float(v) for v in b])

    def slack(self, v):
        s = self.Aq * fmpq_mat(self.n, 1, v)
        return [s[i, 0] - self.bq[i] for i in range(self.m)]

    def vertex_from(self, yf):
        """Exact vertex near the float point yf (from the LP), or None."""
        sl = self.Af @ yf - self.bf
        order = np.argsort(np.abs(sl) / np.maximum(1, np.abs(self.bf)))
        sel = []
        for i in order[:40]:
            if np.linalg.matrix_rank(self.Af[sel + [i]], tol=1e-10) == len(sel) + 1:
                sel.append(int(i))
            if len(sel) == self.n:
                break
        if len(sel) < self.n:
            return None
        B = fmpq_mat(self.n, self.n, [v for i in sel for v in self.A[i]])
        if B.rank() < self.n:
            return None
        v = B.solve(fmpq_mat(self.n, 1, [self.bq[i] for i in sel]))
        v = [v[i, 0] for i in range(self.n)]
        return v if min(self.slack(v)) >= 0 else None

    def start(self, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(20):
            r = linprog(rng.standard_normal(self.n), A_ub=-self.Af, b_ub=-self.bf, bounds=[(None, None)] * self.n,
                        method="highs-ds")
            if r.status == 2:
                return "infeasible"
            if r.status == 0:
                v = self.vertex_from(r.x)
                if v is not None:
                    return v
        return None

    def edges(self, v, T):
        """Exact edge directions at vertex v with tight set T."""
        AT = [self.A[i] for i in T]
        dirs = []
        if len(T) == self.n:
            B = fmpq_mat(self.n, self.n, [x for r in AT for x in r])
            Bi = B.inv()
            return [[Bi[r, c] for r in range(self.n)] for c in range(self.n)]
        MT = fmpq_mat(len(T), self.n, [x for r in AT for x in r])
        seen = set()
        for S in itertools.combinations(range(len(T)), self.n - 1):
            M = fmpq_mat(self.n - 1, self.n, [x for i in S for x in AT[i]])
            R, rk = M.rref()
            if rk != self.n - 1:
                continue
            # null vector from rref: one free column
            piv = []
            r = 0
            for c in range(self.n):
                if r < rk and R[r, c] != 0:
                    piv.append(c); r += 1
            free = [c for c in range(self.n) if c not in piv][0]
            d = [fmpq(0)] * self.n; d[free] = fmpq(1)
            for k, c in enumerate(piv):
                d[c] = -R[k, free]
            Md = MT * fmpq_mat(self.n, 1, d)
            vals = [Md[i, 0] for i in range(len(T))]
            if all(x >= 0 for x in vals):
                s = d
            elif all(x <= 0 for x in vals):
                s = [-x for x in d]
            else:
                continue
            m = max(abs(x) for x in s); key = tuple(x / m for x in s)
            if key not in seen:
                seen.add(key); dirs.append(s)
        return dirs

    def enumerate(self, v0, max_vertices=20000):
        key = lambda v: tuple(v)  # noqa: E731
        V = [v0]; seen = {key(v0): 0}; inc = []; qi = 0
        while qi < len(V):
            v = V[qi]; qi += 1
            s = self.slack(v)
            T = [i for i in range(self.m) if s[i] == 0]
            inc.append(frozenset(T))
            for d in self.edges(v, T):
                Ad = self.Aq * fmpq_mat(self.n, 1, d)
                lam = None
                for i in range(self.m):
                    a = Ad[i, 0]
                    if a < 0 and i not in T:
                        t = s[i] / (-a)
                        if lam is None or t < lam:
                            lam = t
                if lam is None:
                    raise ValueError("unbounded edge (Q must be bounded by N)")
                w = [v[k] + lam * d[k] for k in range(self.n)]
                kw = key(w)
                if kw not in seen:
                    seen[kw] = len(V); V.append(w)
                    if len(V) > max_vertices:
                        raise RuntimeError("vertex budget")
        return V, inc


def irredundant(A, b, tol=1e-9):
    """Float proposal of the irredundant rows (min a_i.y over the other rows < b_i + tol keeps row i).
    Dropping rows only enlarges Q; mass_bounds re-checks every dropped row exactly at every vertex."""
    Af = np.array([[float(v) for v in r] for r in A]); bf = np.array([float(v) for v in b])
    keep = []
    for i in range(len(A)):
        o = np.arange(len(A)) != i
        r = linprog(Af[i], A_ub=-Af[o], b_ub=-bf[o], bounds=[(-NORM - 2, NORM + 2)] * Af.shape[1], method="highs")
        if r.status != 0 or r.fun < bf[i] + tol * max(1, abs(bf[i])):
            keep.append(i)
    return keep


def affine_rank(P):
    if len(P) <= 1:
        return 0
    n = len(P[0])
    M = fmpq_mat(len(P) - 1, n, [P[i][k] - P[0][k] for i in range(1, len(P)) for k in range(n)])
    return M.rank()


def pulling_triangulation(V, inc, n):
    """Exact pulling triangulation of conv(V) from exact incidences. Returns list of vertex-index tuples."""
    rank_cache = {}

    def dim(F):
        if F not in rank_cache:
            rank_cache[F] = affine_rank([V[i] for i in sorted(F)])
        return rank_cache[F]

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
                if len(G) >= d and v0 not in G and G != F and G not in facets and dim(G) == d - 1:
                    facets.add(G)
            # keep only maximal sets (a row may cut a lower face)
            res = []
            for G in facets:
                for s in tri(G, d - 1):
                    res.append((v0,) + s)
        memo[(F, d)] = res
        return res

    F = frozenset(range(len(V)))
    d = dim(F)
    if d < n:
        return [], d
    return tri(F, n), d


# ---------------------------------------------------------------- integrand bounds (arb)
FACT9 = math.factorial(9)


def F_arb(s, t0):
    """(12/pi^5) s^-5 S10(t0 sqrt(s)), S10(r) = e^{-r^2/2} sum_{k<5} (r^2/2)^k/k!; decreasing in s >= 0."""
    s = arb(s); u = arb(fmpq(*t0.as_integer_ratio())) ** 2 / 2 * s
    poly = sum(u ** k / math.factorial(k) for k in range(5))
    return 12 / arb.pi() ** 5 * s ** -5 * (-u).exp() * poly


def sq(x):
    return sum(v * v for v in x)


def min_sq_lower(P):
    """Rigorous lower bound on min |p|^2 over conv(P) (P exact points in R^10): separating hyperplane through a
    float projection point w, min_v w.v / |w| exact; also |m|^2 >= 1/2 on the slice (m_p2 - m_p3 = 1)."""
    Pf = np.array([[float(v) for v in p] for p in P])
    rho = 1e3
    beta, _ = nnls(np.vstack([Pf.T, rho * np.ones(len(P))]), np.r_[np.zeros(Pf.shape[1]), rho])
    w = Pf.T @ beta
    best = fmpq(1, 2)
    if np.linalg.norm(w) > 0:
        wq = [q(v) for v in w]
        L = min(sum(a * b for a, b in zip(wq, p)) for p in P)
        if L > 0:
            best = max(best, L * L / sq(wq))
    return best


def mass_bounds(tri, gates, t0=1.5, tol=1.05, budget=20000, return_info=False):
    """Method M1. Returns (lb, ub) as arb-derived floats rounded outward (ub up, lb down) and info."""
    rows, _ = exact_rows(tri, np.asarray(gates))
    sp = slice_polytope(tri, rows)
    info = {}
    if sp is None:
        return (0.0, 0.0, dict(empty="trivial")) if return_info else (0.0, 0.0)
    A, b, idx = sp
    keep = irredundant(A, b)
    Afull, bfull = A, b
    A = [A[i] for i in keep]; b = [b[i] for i in keep]
    P = Poly(A, b)
    v0 = P.start()
    if v0 == "infeasible" or v0 is None:
        # TODO: rigorous emptiness via safe phase-1 bound; until then fall back to the box bound
        ub = mass_box(tri, gates, t0, sp=sp)
        info.update(empty=str(v0), fallback="M0")
        return (0.0, ub, info) if return_info else (0.0, ub)
    V, inc = P.enumerate(v0)
    # exact check that the dropped rows hold at every vertex: then conv(V) = Q exactly
    drop = [i for i in range(len(Afull)) if i not in set(keep)]
    if drop:
        Ad = fmpq_mat(len(drop), P.n, [v for i in drop for v in Afull[i]])
        for v in V:
            s = Ad * fmpq_mat(P.n, 1, v)
            if any(s[k, 0] < bfull[i] for k, i in enumerate(drop)):
                raise RuntimeError("a dropped row is violated at a vertex: redundancy test wrong")
    S, d = pulling_triangulation(V, inc, P.n)
    info.update(rows=P.m, vertices=len(V), simplices0=len(S), dim=d)
    if d < P.n:
        return (0.0, 0.0, info) if return_info else (0.0, 0.0)
    M = [m_of(tri, idx, v) for v in V]

    def bounds(Ms, vol):
        hi = F_arb(min_sq_lower(Ms), t0) * vol
        lo = F_arb(max(sq(p) for p in Ms), t0) * vol
        return lo, hi

    heap = []; tot_lo = arb(0); tot_hi = arb(0); cnt = 0
    for s in S:
        Ys = [V[i] for i in s]
        det = fmpq_mat(P.n, P.n, [Ys[i][k] - Ys[0][k] for i in range(1, P.n + 1) for k in range(P.n)]).det()
        vol = abs(det) / FACT9
        Ms = [M[i] for i in s]
        lo, hi = bounds(Ms, vol)
        tot_lo += lo; tot_hi += hi; cnt += 1
        heapq.heappush(heap, (-float((hi - lo).upper()), cnt, Ys, Ms, vol, lo, hi))
    while heap and float(tot_hi.upper()) > tol * float(tot_lo.lower()) and len(heap) < budget:
        _, _, Ys, Ms, vol, lo, hi = heapq.heappop(heap)
        tot_lo -= lo; tot_hi -= hi
        # bisect the longest edge (in m-space)
        i, j = max(itertools.combinations(range(len(Ms)), 2), key=lambda e: float(sq([a - b for a, b in zip(Ms[e[0]], Ms[e[1]])])))
        ym = [(a + b) / 2 for a, b in zip(Ys[i], Ys[j])]; mm = [(a + b) / 2 for a, b in zip(Ms[i], Ms[j])]
        for k in (i, j):
            Yc = list(Ys); Mc = list(Ms); Yc[k] = ym; Mc[k] = mm
            lo2, hi2 = bounds(Mc, vol / 2)
            tot_lo += lo2; tot_hi += hi2; cnt += 1
            heapq.heappush(heap, (-float((hi2 - lo2).upper()), cnt, Yc, Mc, vol / 2, lo2, hi2))
    info.update(simplices=len(heap))
    ub = float(np.nextafter(float(tot_hi.upper()), np.inf)); lb = max(0.0, float(np.nextafter(float(tot_lo.lower()), -np.inf)))
    return (lb, ub, info) if return_info else (lb, ub)


# ---------------------------------------------------------------- M0: bounding box of the slice
def safe_min_exact(c, A, b, lo, hi, y):
    """Rigorous lower bound on min c.v s.t. A v >= b, lo <= v <= hi (all exact), for any float y >= 0."""
    yq = [q(max(0.0, v)) for v in y]
    r = list(c)
    val = fmpq(0)
    for i, yi in enumerate(yq):
        if yi != 0:
            val += yi * b[i]
            for k in range(len(r)):
                r[k] -= yi * A[i][k]
    for k in range(len(r)):
        val += min(r[k] * lo[k], r[k] * hi[k])
    return val


def mass_box(tri, gates, t0=1.5, sp=None):
    if sp is None:
        rows, _ = exact_rows(tri, np.asarray(gates))
        sp = slice_polytope(tri, rows)
        if sp is None:
            return 0.0
    A, b, idx = sp
    n = len(A[0]); Af = np.array([[float(v) for v in r] for r in A]); bf = np.array([float(v) for v in b])
    ip3 = idx.index(tri[2])
    lo = [fmpq(-NORM) for _ in range(n)]; hi = [fmpq(NORM) for _ in range(n)]
    lo[ip3] = fmpq(-NORM - 1); hi[ip3] = fmpq(NORM - 1)
    bl, bh = list(lo), list(hi)
    for k in range(n):
        for sgn in (1, -1):
            c = np.zeros(n); c[k] = sgn
            r = linprog(c, A_ub=-Af, b_ub=-bf, bounds=list(zip(map(float, lo), map(float, hi))), method="highs")
            if r.status == 2:
                return 0.0   # TODO rigorous emptiness (phase-1); M0 treats float infeasibility as unproven -> skip
            if r.status != 0:
                continue
            y = -r.ineqlin.marginals
            cq = [fmpq(sgn) if i == k else fmpq(0) for i in range(n)]
            val = safe_min_exact(cq, A, b, lo, hi, y)
            if sgn == 1:
                bl[k] = max(bl[k], val)
            else:
                bh[k] = min(bh[k], -val)
    vol = fmpq(1)
    for k in range(n):
        if bh[k] < bl[k]:
            return 0.0
        vol *= bh[k] - bl[k]
    # min |m|^2 over the box (separable): m_j = y_j, plus m_p2 = 1 + y_p3
    def mn(l, h, off=0):
        l, h = l + off, h + off
        return fmpq(0) if l <= 0 <= h else min(l * l, h * h)
    s = sum(mn(bl[k], bh[k]) for k in range(n)) + mn(bl[ip3], bh[ip3], 1)
    s = max(s, fmpq(1, 2))
    return float(np.nextafter(float((F_arb(s, t0) * vol).upper()), np.inf))


def norm_loss(t0=1.5):
    """Part (c): P(E(t0) \\ N) <= P(||x||_inf > NORM t0) <= 20 phi(NORM t0)/(NORM t0) (Mills ratio)."""
    r = arb(fmpq(*t0.as_integer_ratio())) * NORM
    return 20 * (-(r * r) / 2).exp() / (2 * arb.pi()).sqrt() / r


# ---------------------------------------------------------------- M2: active-set orthant bound (Gaussian tilt)
def event_rows(tri, gates, t0=1.5):
    """Exact rows (a, b) with a.x >= b for P = region ∩ pick p3 ∩ order ∩ N ∩ {g23 >= t0} in R^10,
    each row scaled to max |a_k| = 1, duplicates merged (tightest kept). None if trivially empty."""
    p1, p2, p3 = tri
    rows, _ = exact_rows(tri, np.asarray(gates))
    zero = fmpq(0)
    g = [zero] * D; g[p2] = fmpq(1); g[p3] = fmpq(-1)
    full = [(r, zero) for r in rows]
    for j in range(D):                                  # N: NORM g23 -+ x_j >= 0
        for s in (1, -1):
            a = [NORM * v for v in g]; a[j] -= s
            full.append((a, zero))
    full.append((g, fmpq(*t0.as_integer_ratio())))
    out = {}
    for a, b in full:
        m = max(abs(v) for v in a)
        if m == 0:
            if b > 0:
                return None
            continue
        key = tuple(v / m for v in a); b = b / m
        if key not in out or b > out[key]:
            out[key] = b
    return [list(k) for k in out], [out[k] for k in out]


def _Mbound(mu, Dj):
    """∫_0^∞ exp(-mu t - Dj t^2 / 2) dt = sqrt(2 pi / Dj) exp(mu^2/(2 Dj)) Phi(-mu/sqrt(Dj)) (arb)."""
    mu, Dj = arb(mu), arb(Dj)
    z = mu / Dj.sqrt()
    return (2 * arb.pi() / Dj).sqrt() * (z * z / 2).exp() * (z / arb(2).sqrt()).erfc() / 2


def orthant_bound(A, b, S):
    """Rigorous upper bound on P(A_S X >= b_S), X ~ N(0, I_10): Y = A_S X ~ N(0, Sig), tilt at c = b_S:
    P = phi_Sig(c) ∫_{t>=0} exp(-mu.t - t' Sig^-1 t / 2), mu = Sig^-1 c, and t' Sig^-1 t >= sum D_j t_j^2 on t >= 0
    with D_j = (Sig^-1)_jj - sum_{i != j} max(0, -(Sig^-1)_ij) (needs D_j > 0). Returns arb or None."""
    k = len(S)
    AS = fmpq_mat(k, D, [v for i in S for v in A[i]])
    Sig = AS * AS.transpose()
    det = Sig.det()
    if det == 0:
        return None
    Si = Sig.inv()
    c = fmpq_mat(k, 1, [b[i] for i in S])
    mu = Si * c
    quad = (c.transpose() * mu)[0, 0]
    tot = (2 * arb.pi()) ** (-arb(k) / 2) / arb(det).sqrt() * (-arb(quad) / 2).exp()
    for j in range(k):
        Dj = Si[j, j] - sum((-Si[i, j] for i in range(k) if i != j and Si[i, j] < 0), fmpq(0))
        if Dj <= 0:
            return None
        tot *= _Mbound(mu[j, 0], Dj)
    return tot


def dominating_point(Af, bf):
    """Float argmin |x|^2 s.t. Af x >= bf (SLSQP), as a proposal only."""
    from scipy.optimize import minimize
    x0 = linprog(np.zeros(D), A_ub=-Af, b_ub=-bf, bounds=[(None, None)] * D, method="highs").x
    r = minimize(lambda x: x @ x, x0, jac=lambda x: 2 * x, method="SLSQP",
                 constraints=[dict(type="ineq", fun=lambda x: Af @ x - bf, jac=lambda x: Af)],
                 options=dict(maxiter=500, ftol=1e-14))
    return r.x


def mass_orthant(tri, gates, t0=1.5, kmax=10, return_info=False):
    """Method M2: min over greedily grown row sets S (by slack at the dominating point) of orthant_bound.
    Rigorous for every S; floats only choose S."""
    er = event_rows(tri, gates, t0)
    if er is None:
        return (0.0, {}) if return_info else 0.0
    A, b = er
    Af = np.array([[float(v) for v in r] for r in A]); bf = np.array([float(v) for v in b])
    if linprog(np.zeros(D), A_ub=-Af, b_ub=-bf, bounds=[(None, None)] * D, method="highs").status == 2:
        # float-infeasible: TODO rigorous emptiness certificate; fall through and bound anyway
        pass
    x = dominating_point(Af, bf)
    sl = (Af @ x - bf) / np.linalg.norm(Af, axis=1)
    order = list(np.argsort(sl))
    best = None; S = []; bestS = None
    for i in order:
        if len(S) == kmax:
            break
        T = S + [int(i)]
        if np.linalg.matrix_rank(Af[T], tol=1e-9) < len(T):
            continue
        v = orthant_bound(A, b, T)
        if v is None:
            continue
        S = T
        if best is None or float(v.upper()) < float(best.upper()):
            best, bestS = v, list(T)
    ub = float(np.nextafter(float(best.upper()), np.inf)) if best is not None else 1.0
    info = dict(r=float(np.linalg.norm(x)), k=len(bestS or []), slack=[float(sl[i]) for i in (bestS or [])])
    return (ub, info) if return_info else ub


# ---------------------------------------------------------------- M3: exact volume distribution (Lawrence) bound
def slice_rows_with_norm(tri, gates, NORM_=NORM):
    rows, _ = exact_rows(tri, np.asarray(gates))
    return slice_polytope(tri, rows, NORM=NORM_)


def proj_point(Af, bf, ip3):
    """Float minimiser of s(y) = |y|^2 + (1 + y_p3)^2 over {Af y >= bf} (proposal only)."""
    from scipy.optimize import minimize
    n = Af.shape[1]; G = np.eye(n); G[ip3, ip3] = 2; y0 = np.zeros(n); y0[ip3] = -.5
    x0 = linprog(np.zeros(n), A_ub=-Af, b_ub=-bf, bounds=[(None, None)] * n, method="highs").x
    r = minimize(lambda y: (y - y0) @ G @ (y - y0), x0, jac=lambda y: 2 * G @ (y - y0), method="SLSQP",
                 constraints=[dict(type="ineq", fun=lambda y: Af @ y - bf, jac=lambda y: Af)],
                 options=dict(maxiter=500, ftol=1e-15))
    return r.x


def mass_lawrence(tri, gates, t0=1.5, NORM_=NORM, ratio=1.01, prec=512, return_info=False, poly=None):
    """Method M3. On the slice, s(y) = |m(y)|^2 >= sigma(tau) := s(c) + tau + kappa tau^2 with tau = l.(y - c),
    l = grad s(c), kappa = 1/(l' G^-1 l) (exact identity + Cauchy-Schwarz), for ANY rational c. F is decreasing,
    so mass <= (12/pi^5-free F) ∫ F(sigma(tau)) dV(tau), V(T) = vol{y in Q: tau(y) <= T}. For a simple polytope V is
    Lawrence's piecewise polynomial V(T) = sum_v w_v (T - tau_v)_+^9 / 9!, w_v = |det A_v|^-1 / prod_i (l.d_i),
    evaluated exactly (vertices, edges) and in arb. Upper Riemann sum on a grid where F changes by <= ratio."""
    import flint
    old = flint.ctx.prec; flint.ctx.prec = prec
    try:
        sp = slice_rows_with_norm(tri, gates, NORM_)
        if sp is None:
            return (0.0, dict(empty="trivial")) if return_info else 0.0
        A, b, idx = sp
        keep = irredundant(A, b)
        Afull, bfull = A, b
        A = [A[i] for i in keep]; b = [b[i] for i in keep]
        P = Poly(A, b) if poly is None else poly
        v0 = P.start()
        if v0 == "infeasible" or v0 is None:
            raise RuntimeError("no start vertex (emptiness certificate not implemented here)")
        V, inc = P.enumerate(v0)
        drop = [i for i in range(len(Afull)) if i not in set(keep)]
        if drop:
            Ad = fmpq_mat(len(drop), P.n, [v for i in drop for v in Afull[i]])
            for v in V:
                s = Ad * fmpq_mat(P.n, 1, v)
                if any(s[k, 0] < bfull[i] for k, i in enumerate(drop)):
                    raise RuntimeError("dropped row violated")
        if any(len(T) != P.n for T in inc):
            raise RuntimeError("degenerate vertex: Lawrence formula needs a simple polytope")
        n = P.n; ip3 = idx.index(tri[2])
        cf = proj_point(P.Af, P.bf, ip3)
        for attempt in range(5):
            c = [q(v) for v in cf]
            y0 = [fmpq(0)] * n; y0[ip3] = fmpq(-1, 2)
            Gd = [fmpq(1)] * n; Gd[ip3] = fmpq(2)
            l = [2 * Gd[k] * (c[k] - y0[k]) for k in range(n)]
            sc = sum(Gd[k] * (c[k] - y0[k]) ** 2 for k in range(n)) + fmpq(1, 2)
            kappa = 1 / sum(l[k] ** 2 / Gd[k] for k in range(n))
            lc = sum(l[k] * c[k] for k in range(n))
            Lq = fmpq_mat(1, n, l)
            taus, ws = [], []
            ok = True
            for v, T in zip(V, inc):
                T = sorted(T)
                B = fmpq_mat(n, n, [x for i in T for x in A[i]])
                Bi = B.inv()
                ld = Lq * Bi                      # l.d_i for each edge i (columns of Bi)
                prod = fmpq(1)
                for i in range(n):
                    if ld[0, i] == 0:
                        ok = False; break
                    prod *= ld[0, i]
                if not ok:
                    break
                taus.append(sum(l[k] * v[k] for k in range(n)) - lc)
                ws.append(1 / (abs(B.det()) * prod))
            if ok:
                break
            cf = cf + 1e-7 * np.random.default_rng(attempt).standard_normal(n)
        else:
            raise RuntimeError("non-generic direction")
        tw = sorted(zip(taus, ws))
        tmin, tmax = tw[0][0], tw[-1][0]
        tA = [arb(t) for t, _ in tw]; wA = [arb(w) / FACT9 for _, w in tw]

        def Vol(Tq):
            T = arb(Tq); tot = arb(0)
            for t, w, (tq, _) in zip(tA, wA, tw):
                if tq < Tq:
                    tot += w * (T - t) ** 9
            return tot

        def sigma_min(a, bb):                  # min of sigma on [a, bb]; sigma minimised at -1/(2 kappa)
            m = -1 / (2 * kappa)
            t = a if m <= a else (bb if m >= bb else m)
            return max(sc + t + kappa * t * t, fmpq(1, 2))

        # grid: start at tmin, step so that F(sigma) drops by <= ratio, up to tmax
        grid = [tmin]; Fv = []
        t = tmin
        while t < tmax:
            # step: d log F / ds <= ~ (5/s + a); choose dt so (dsigma)*(5/s + a + 1) <= log(ratio)
            s = float(sigma_min(t, t))
            dlog = 5 / s + t0 * t0 / 2 + 1
            dsig = (1 + 2 * float(kappa) * max(float(t), 0)) + 1e-12
            dt = math.log(ratio) / (dlog * dsig)
            t = min(tmax, t + q(max(dt, 1e-9 * (float(tmax) - float(tmin)))))
            grid.append(t)
        tot = arb(0); prev = arb(0); vols = []
        for j in range(1, len(grid)):
            cur = Vol(grid[j])
            inc_ = cur - prev
            up = inc_.upper() if inc_.upper() > 0 else arb(0)
            tot += F_arb(sigma_min(grid[j - 1], grid[j]), t0) * up
            prev = cur
        info = dict(vertices=len(V), grid=len(grid), vol=float(prev.mid()), vol_rad=float(prev.rad()),
                    s_c=float(sc), tau_range=(float(tmin), float(tmax)))
        ub = float(np.nextafter(float(tot.upper()), np.inf))
        return (ub, info) if return_info else ub
    finally:
        flint.ctx.prec = old


# ---------------------------------------------------------------- M4: M3 on an adaptive exact partition of Q
class Piece:
    """A simple polytope {y: A_i.y >= b_i, i in rows} given by its exact vertices and tight sets (9 rows each).
    A, b are shared, growing lists (cuts append rows)."""

    def __init__(self, A, b, rows, V, T):
        self.A, self.b, self.rows, self.V, self.T = A, b, rows, V, T
        self.ub = None

    def edges(self):
        idx = {}
        for i, T in enumerate(self.T):
            for drop in T:
                key = T - {drop}
                idx.setdefault(key, []).append(i)
        return [tuple(v) for v in idx.values() if len(v) == 2]

    def cut(self, h, beta):
        """Split by h.y >= beta (exact): returns (piece >=, piece <=). beta must avoid every vertex value."""
        A, b = self.A, self.b
        r_ge = len(A); A.append(list(h)); b.append(beta)
        r_le = len(A); A.append([-v for v in h]); b.append(-beta)
        val = [sum(x * y for x, y in zip(h, v)) - beta for v in self.V]
        if any(x == 0 for x in val):
            raise ValueError("cut through a vertex")
        up = [i for i, x in enumerate(val) if x > 0]; dn = [i for i, x in enumerate(val) if x < 0]
        newV, newT_up, newT_dn = [], [], []
        for i, j in self.edges():
            if (val[i] > 0) != (val[j] > 0):
                lam = val[i] / (val[i] - val[j])
                w = [a + lam * (c - a) for a, c in zip(self.V[i], self.V[j])]
                common_ = self.T[i] & self.T[j]
                newV.append(w); newT_up.append(common_ | {r_ge}); newT_dn.append(common_ | {r_le})
        P1 = Piece(A, b, self.rows + [r_ge], [self.V[i] for i in up] + newV, [self.T[i] for i in up] + newT_up)
        P2 = Piece(A, b, self.rows + [r_le], [self.V[i] for i in dn] + newV, [self.T[i] for i in dn] + newT_dn)
        return P1, P2


_binv_cache = {}


def _vertex_data(A, T):
    key = (id(A), tuple(sorted(T)))
    if key not in _binv_cache:
        Ts = sorted(T)
        B = fmpq_mat(len(Ts), len(Ts), [x for i in Ts for x in A[i]])
        _binv_cache[key] = (B.inv(), abs(B.det()))
    return _binv_cache[key]


def lawrence_bound(piece, tri, idx, t0, ratio=1.02, cf=None):
    """M3 bound on one piece (rigorous for any rational c). Exact cumulative polynomial for V(T)."""
    A, n = piece.A, len(piece.A[0])
    ip3 = idx.index(tri[2])
    rows = piece.rows
    Af = np.array([[float(v) for v in A[i]] for i in rows]); bf = np.array([float(piece.b[i]) for i in rows])
    if cf is None:
        cf = proj_point(Af, bf, ip3)
    rng = np.random.default_rng(0)
    for attempt in range(6):
        c = [q(v) for v in cf]
        y0 = [fmpq(0)] * n; y0[ip3] = fmpq(-1, 2)
        Gd = [fmpq(1)] * n; Gd[ip3] = fmpq(2)
        l = [2 * Gd[k] * (c[k] - y0[k]) for k in range(n)]
        sc = sum(Gd[k] * (c[k] - y0[k]) ** 2 for k in range(n)) + fmpq(1, 2)
        kappa = 1 / sum(l[k] ** 2 / Gd[k] for k in range(n))
        lc = sum(l[k] * c[k] for k in range(n))
        Lq = fmpq_mat(1, n, l)
        tw = []; ok = True
        for v, T in zip(piece.V, piece.T):
            Bi, det = _vertex_data(A, T)
            ld = Lq * Bi
            prod = fmpq(1)
            for i in range(n):
                if ld[0, i] == 0:
                    ok = False; break
                prod *= ld[0, i]
            if not ok:
                break
            tw.append((sum(l[k] * v[k] for k in range(n)) - lc, 1 / (det * prod)))
        if ok:
            break
        cf = cf + 1e-7 * rng.standard_normal(n)
    else:
        raise RuntimeError("non-generic direction")
    tw.sort()
    tmin, tmax = tw[0][0], tw[-1][0]
    # exact cumulative polynomial: on [tau_j, tau_{j+1}), 9! V(T) = sum_{v: tau_v <= tau_j} w_v (T - tau_v)^9
    from math import comb
    coef = [fmpq(0)] * 10                    # coefficients of T^k
    bps = []                                  # (tau, coef snapshot after adding all vertices at tau)
    for tv, w in tw:
        p = fmpq(1)
        mt = -tv
        pw = [fmpq(1)]
        for _ in range(9):
            pw.append(pw[-1] * mt)
        for kk in range(10):
            coef[kk] += w * comb(9, kk) * pw[9 - kk]
        bps.append((tv, list(coef)))

    def Vexact(Tq):
        # last breakpoint with tau <= Tq
        lo_, hi_ = 0, len(bps) - 1
        if Tq < bps[0][0]:
            return fmpq(0)
        while lo_ < hi_:
            mid = (lo_ + hi_ + 1) // 2
            if bps[mid][0] <= Tq:
                lo_ = mid
            else:
                hi_ = mid - 1
        cf_ = bps[lo_][1]
        tot = fmpq(0)
        for kk in range(9, -1, -1):
            tot = tot * Tq + cf_[kk]
        return tot / FACT9

    def sigma_min(a, bb):
        m = -1 / (2 * kappa)
        t = a if m <= a else (bb if m >= bb else m)
        return max(sc + t + kappa * t * t, fmpq(1, 2))

    grid = [tmin]; t = tmin
    while t < tmax:
        s = float(sigma_min(t, t))
        dlog = 5 / s + t0 * t0 / 2 + 1
        dsig = 1 + 2 * float(kappa) * abs(float(t)) + 1e-12
        dt = math.log(ratio) / (dlog * dsig)
        t = min(tmax, t + q(max(dt, 1e-9 * (float(tmax) - float(tmin)) + 1e-300)))
        grid.append(t)
    tot = arb(0); prev = fmpq(0)
    for j in range(1, len(grid)):
        cur = Vexact(grid[j])
        incv = cur - prev
        if incv > 0:
            tot += F_arb(sigma_min(grid[j - 1], grid[j]), t0) * arb(incv)
        prev = cur
    piece.ub = tot
    piece.sc = float(sc)
    piece.vol = prev
    return tot


def mass_split(tri, gates, t0=1.5, NORM_=NORM, max_pieces=64, rel_stop=0.02, ratio=1.02, return_info=False):
    """Method M4: M3 on an adaptive exact partition of the slice polytope (cuts through the piece's vertex
    spread, largest-bound piece first). Rigorous: pieces tile Q exactly; each bound is rigorous."""
    sp = slice_rows_with_norm(tri, gates, NORM_)
    if sp is None:
        return (0.0, {}) if return_info else 0.0
    A, b, idx = sp
    keep = irredundant(A, b)
    Afull, bfull = A, b
    A = [A[i] for i in keep]; b = [b[i] for i in keep]
    P = Poly(A, b); v0 = P.start()
    if v0 == "infeasible" or v0 is None:
        raise RuntimeError("no start vertex")
    V, inc = P.enumerate(v0)
    drop = [i for i in range(len(Afull)) if i not in set(keep)]
    if drop:
        Ad = fmpq_mat(len(drop), P.n, [v for i in drop for v in Afull[i]])
        for v in V:
            s = Ad * fmpq_mat(P.n, 1, v)
            if any(s[k, 0] < bfull[i] for k, i in enumerate(drop)):
                raise RuntimeError("dropped row violated")
    if any(len(T) != P.n for T in inc):
        raise RuntimeError("degenerate vertex")
    A = list(A); b = list(b)
    root = Piece(A, b, list(range(len(A))), V, [frozenset(T) for T in inc])
    lawrence_bound(root, tri, idx, t0, ratio)
    pieces = [root]; hist = [float(root.ub.upper())]
    ip3 = idx.index(tri[2])
    while len(pieces) < max_pieces:
        pieces.sort(key=lambda p: -float(p.ub.upper()))
        top = pieces[0]
        Vf = np.array([[float(x) for x in v] for v in top.V])
        Mf = Vf.copy(); Mf[:, ip3] *= np.sqrt(2)          # G-metric coordinates (approx; only for the choice)
        cov = np.cov((Mf - Mf.mean(0)).T)
        w, E = np.linalg.eigh(cov)
        hdir = E[:, -1].copy(); hdir[ip3] *= np.sqrt(2)     # normal in y coordinates
        proj = Vf @ hdir
        beta = float(np.median(proj))
        h = [q(round(v, 12)) for v in hdir]
        bq = q(beta) + fmpq(1, 10 ** 13 + 7)
        try:
            p1, p2 = top.cut(h, bq)
        except ValueError:
            bq += fmpq(1, 10 ** 11 + 3); p1, p2 = top.cut(h, bq)
        if not p1.V or not p2.V:
            break
        for p in (p1, p2):
            lawrence_bound(p, tri, idx, t0, ratio)
        pieces = pieces[1:] + [p1, p2]
        tot = sum(float(p.ub.upper()) for p in pieces)
        hist.append(tot)
        if len(hist) > 8 and hist[-9] - tot < rel_stop * tot:
            break
    total = arb(0)
    for p in pieces:
        total += p.ub
    ub = float(np.nextafter(float(total.upper()), np.inf))
    info = dict(vertices=len(V), pieces=len(pieces), hist=hist[:: max(1, len(hist) // 8)])
    return (ub, info) if return_info else ub


# ---------------------------------------------------------------- cheap call for unfinished cells (charge and stop)
def _orthant_float(Af, bf, S):
    """Float version of orthant_bound (used only to choose S)."""
    from scipy.stats import norm
    A = Af[S]; Sig = A @ A.T
    try:
        M = np.linalg.inv(Sig)
    except np.linalg.LinAlgError:
        return np.inf
    c = bf[S]; mu = M @ c; k = len(S)
    off = ~np.eye(k, dtype=bool)
    Dd = np.diag(M) - np.where(off, np.maximum(0, -M), 0).sum(0)
    if (Dd <= 0).any():
        return np.inf
    z = mu / np.sqrt(Dd)
    lM = 0.5 * np.log(2 * np.pi / Dd) + norm.logsf(z) + z * z / 2
    return np.exp(-k / 2 * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(Sig)[1] - c @ mu / 2 + lM.sum())


def cell_rows_exact(tri, gates, k, t0, event=False, norm=True, normv=NORM):
    """Exact (a, b) rows, a.x >= b: cell gates of steps 1..k, ordering, g23 >= t0, optionally N and the event."""
    g = np.array(gates).copy(); g[k:] = -1
    if event:
        assert k == D
        g = np.array(gates)
    rows, _ = exact_rows(tri, g, pick=event)
    p2, p3 = tri[1], tri[2]
    zero = fmpq(0)
    gg = [zero] * D; gg[p2] = fmpq(1); gg[p3] = fmpq(-1)
    full = [(r, zero) for r in rows]
    if norm:
        for j in range(D):
            for s in (1, -1):
                a = [normv * v for v in gg]; a[j] -= s
                full.append((a, zero))
    full.append((gg, fmpq(*float(t0).as_integer_ratio())))
    out = {}
    for a, b in full:
        m = max(abs(v) for v in a)
        if m == 0:
            if b > 0:
                return None
            continue
        key = tuple(v / m for v in a); b = b / m
        if key not in out or b > out[key]:
            out[key] = b
    return [list(kk) for kk in out], [out[kk] for kk in out]


def mass_upper(tri, gates, k, t0=1.5, kmax=10, event=None, return_info=False, norm=NORM):
    """Cheap rigorous upper bound on gamma(cell ∩ N ∩ {g23 >= t0}) (k < 10, event ignored) or, for k = 10,
    on gamma(region ∩ N ∩ E(t0)) (event=True default). Method M2: P(A_S X >= b_S) for a row subset S
    (|S| <= 10) chosen greedily in float; the bound for the chosen S is computed exactly (fmpq + arb)."""
    if event is None:
        event = (k == D)
    er = cell_rows_exact(tri, gates, k, t0, event=event, normv=norm)
    if er is None:
        return (0.0, {}) if return_info else 0.0
    A, b = er
    Af = np.array([[float(v) for v in r] for r in A]); bf = np.array([float(v) for v in b])
    x = dominating_point(Af, bf)
    sl = (Af @ x - bf) / np.linalg.norm(Af, axis=1)
    cand = [int(i) for i in np.argsort(sl)[:30]]
    S = []; best = np.inf
    for _ in range(kmax):
        trial = []
        for j in cand:
            if j in S:
                continue
            T = S + [j]
            if np.linalg.matrix_rank(Af[T], tol=1e-9) < len(T):
                continue
            trial.append((_orthant_float(Af, bf, T), j))
        if not trial:
            break
        v, j = min(trial)
        if v >= best:
            break
        best = v; S.append(j)
    ex = orthant_bound(A, b, S) if S else None
    ub = float(np.nextafter(float(ex.upper()), np.inf)) if ex is not None else 1.0
    info = dict(r=float(np.linalg.norm(x)), S=len(S), float_est=float(best))
    return (ub, info) if return_info else ub


# ---------------------------------------------------------------- M5: exact marginal reduction for prefix cells
def _ipow(x, m):
    r = arb(1)
    for _ in range(m):
        r *= x
    return r


def _G_arb(d, s, a, b):
    """∫_a^b z^(d-1) exp(-z^2 s / 2) dz (b may be None = inf), s > 0, in arb."""
    s = arb(s); h = arb(d) / 2
    up = lambda z: (arb(0) if z is None else (arb(z) ** 2 * s / 2).gamma_upper(h))  # noqa: E731
    return (2 / s) ** h / 2 * (up(a) - up(b))


def _E_simplex_exp(mu, t, upper):
    """Rigorous bound on E_unif over a simplex of exp(-mu tau), tau affine with vertex values t (fmpq, >= 0),
    mu >= 0 (arb): E = n! h[t_0..t_n] / (-mu)^n, h(t) = exp(-mu t) (Hermite-Genocchi), in arb. Equal knots are
    separated (down for an upper bound, up for a lower bound; E is monotone in each knot). Falls back to the
    trivial bound (1, or exp(-mu max t)) if the arb ball is not tight."""
    n = len(t) - 1
    ts = sorted(t)
    eps = fmpq(1, 10 ** 12) * (1 + max(ts) - min(ts))
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + eps
    if upper:                                            # shifting knots up lowers E: undo by shifting all down
        shift = ts[-1] - max(t)
        ts = [x - shift for x in ts]
    trivial = arb(1) if upper else (-mu * arb(max(t))).exp()
    if mu == 0 or n == 0:
        return trivial
    T = [arb(x) for x in ts]
    dd = [(-mu * x).exp() for x in T]
    for lev in range(1, n + 1):
        dd = [(dd[i + 1] - dd[i]) / (T[i + lev] - T[i]) for i in range(len(dd) - 1)]
    E = dd[0] * math.factorial(n) / (-mu) ** n
    if not E.is_finite() or float(E.rad()) > 1e-6 * abs(float(E.mid())) or float(E.mid()) <= 0:
        return trivial
    return E.upper() if upper else E.lower()


def _E_exp(tv, upper):
    """Rigorous bound on E_unif over a simplex of exp(-u), u affine with vertex values tv (arb balls):
    E = n! h[t_0..t_n] (-1)^n, h = exp(-t) (Hermite-Genocchi). Upper: knots replaced by lower endpoints and moved
    down to separate them (E decreases in every knot); lower: upper endpoints moved up. Falls back to
    exp(-min t) / exp(-max t) when the arb ball is not tight."""
    n = len(tv) - 1
    pts = [x.lower() if upper else x.upper() for x in tv]
    lo_ = min(pts, key=float); hi_ = max(pts, key=float)
    trivial = (-lo_).exp() if upper else (-hi_).exp()
    if n == 0:
        return trivial
    eps = arb(fmpq(1, 10 ** 9)) * (1 + (hi_ - lo_))
    order = sorted(range(len(pts)), key=lambda i: float(pts[i]))
    T = [None] * len(pts)
    for r, i in enumerate(order):
        T[r] = pts[i] - r * eps if upper else pts[i] + r * eps
    dd = [(-x).exp() for x in T]
    for lev in range(1, n + 1):
        dd = [(dd[i + 1] - dd[i]) / (T[i + lev] - T[i]) for i in range(len(dd) - 1)]
    E = dd[0] * math.factorial(n) * (-1) ** n
    if not E.is_finite() or float(E.rad()) > 1e-6 * abs(float(E.mid())) or float(E.mid()) <= 0:
        return trivial
    return E.upper() if upper else E.lower()


def _logPhi(x):
    return ((-x) / arb(2).sqrt()).erfc().log() - arb(2).log()


def _bounds_tangent(Ys, Ms, vol, d, zg, mlow, ip3, c0):
    """Per-simplex bounds for M5t. Exact identity s(y) = s_c + tau(y) + |m(y) - m_c|^2, tau = 2 m_c.(m - m_c)
    affine; upper uses s >= s_c + tau, lower uses s <= s_c + tau + R^2 (R^2 = max vertex |M_i - m_c|^2).
    Phi(z y_p3)^mlow: upper by the tangent of the concave log Phi at x0 = z_mid y_p3(c), lower by its chord over
    the simplex x bin; both are exp-linear in y. The exp-linear part is integrated exactly over the simplex
    (_E_exp); z = g23 is binned with lambda = z^2/2 bounded per bin; the last (unbounded) bin uses the
    s_min / s_max bounds."""
    Pf = np.array([[float(v) for v in p] for p in Ms])
    rho = 1e3
    beta, _ = nnls(np.vstack([Pf.T, rho * np.ones(len(Ms))]), np.r_[np.zeros(Pf.shape[1]), rho])
    bq = [q(v) for v in beta]; tot = sum(bq)
    if tot == 0:
        bq = [fmpq(1, len(Ms))] * len(Ms); tot = fmpq(1)
    bq = [v / tot for v in bq]
    mc = [sum(bq[i] * Ms[i][k] for i in range(len(Ms))) for k in range(len(Ms[0]))]
    sc = sq(mc)
    tau = [2 * sum(a * (b - c) for a, b, c in zip(mc, M, mc)) for M in Ms]
    tmin = min(tau); tsh = [arb(x - tmin) for x in tau]
    R2 = max(sq([b - c for b, c in zip(M, mc)]) for M in Ms)
    smin = max(min_sq_lower(Ms), fmpq(1, 2)); smax = max(sq(p) for p in Ms)
    yp = [y[ip3] for y in Ys]; ypA = [arb(v) for v in yp]
    yp_hi = max(yp); yp_lo = min(yp); ypc = sum(bq[i] * yp[i] for i in range(len(yp)))
    neg = arb(max(fmpq(0), -yp_lo))
    hi = arb(0); lo = arb(0)
    for a, bb in zip(zg[:-1], zg[1:]):
        if bb is None:
            if mlow:
                sup_phi = arb(1) if yp_hi >= 0 else _ipow((-(arb(a) * arb(yp_hi)) / arb(2).sqrt()).erfc() / 2, mlow)
                inf_phi = arb(0) if yp_lo < 0 else _ipow((-(arb(a) * arb(yp_lo)) / arb(2).sqrt()).erfc() / 2, mlow)
            else:
                sup_phi = inf_phi = arb(1)
            hi += _G_arb(d, smin, a, bb) * sup_phi
            lo += _G_arb(d, smax, a, bb) * inf_phi
            continue
        A_, B_ = arb(a), arb(bb)
        llo = A_ ** 2 / 2; lhi = B_ ** 2 / 2
        lx = lhi if tmin < 0 else llo
        ly = llo if tmin < 0 else lhi
        if mlow:
            x0 = (A_ + B_) / 2 * arb(ypc)
            lp0 = _logPhi(x0)
            r0 = (-(x0 * x0) / 2).exp() / (2 * arb.pi()).sqrt() / lp0.exp()
            cu = mlow * (lp0 - r0 * x0 + r0 * (B_ - A_) * neg)
            uu = [llo * t - mlow * r0 * B_ * y for t, y in zip(tsh, ypA)]
            xl = min(a * yp_lo, bb * yp_lo); xh = max(a * yp_hi, bb * yp_hi)
            lxl = _logPhi(arb(xl))
            if xh > xl:
                bet = (_logPhi(arb(xh)) - lxl) / (arb(xh) - arb(xl))
            else:
                bet = arb(0)
            cl = mlow * (lxl - bet * arb(xl) - bet * (B_ - A_) * neg)
            ul = [lhi * t - mlow * bet * A_ * y for t, y in zip(tsh, ypA)]
        else:
            cu = cl = arb(0)
            uu = [llo * t for t in tsh]; ul = [lhi * t for t in tsh]
        Eu = (-lx * arb(tmin) + cu).exp() * _E_exp(uu, True)
        El = (-ly * arb(tmin) + cl).exp() * _E_exp(ul, False)
        if mlow:                                          # also the constant sup/inf of Phi^m; keep the better
            zt = bb if yp_hi >= 0 else a
            sup_phi = _ipow((-(arb(zt) * arb(yp_hi)) / arb(2).sqrt()).erfc() / 2, mlow)
            zl = a if yp_lo >= 0 else bb
            inf_phi = _ipow((-(arb(zl) * arb(yp_lo)) / arb(2).sqrt()).erfc() / 2, mlow)
            Eu2 = (-lx * arb(tmin)).exp() * sup_phi * _E_exp([llo * t for t in tsh], True)
            El2 = (-ly * arb(tmin)).exp() * inf_phi * _E_exp([lhi * t for t in tsh], False)
            if not Eu.is_finite() or float(Eu2.upper()) < float(Eu.upper()):
                Eu = Eu2
            if not El.is_finite() or float(El2.lower()) > float(El.lower()):
                El = El2
        hi += _G_arb(d, sc, a, bb) * Eu
        lo += _G_arb(d, sc + R2, a, bb) * El
    return c0 * vol * lo, c0 * vol * hi


def mass_cell(tri, gates, k, t0=1.5, tol=1.1, budget=4000, zbins=48, return_info=False, tangent=False, zratio=fmpq(27, 25), norm=NORM):
    """Method M5 (k < 10, event ignored): rigorous [lb, ub] on gamma(cell ∩ N_I ∩ {g23 >= t0}).
    The cell's gate rows involve x_0..x_{k-1} only. With I = {0..k-1} ∪ tri, the m lows outside I enter only
    through x_l <= x_p3, i.e. a factor Phi(x_p3)^m (exact). N is kept for the coordinates in I (dropping it
    elsewhere only enlarges the set). The remaining d = |I|-dim integral is done on the slice g23 = 1 (exact
    vertex enumeration + pulling triangulation in d-1 dims), with z = g23 binned, and per (simplex, z-bin)
    sup/inf bounds of the integrand in arb. Refines the simplex with the largest ub - lb."""
    p1, p2, p3 = tri
    I = sorted(set(range(k)) | set(tri)); d = len(I); mlow = D - d
    g = np.array(gates).copy(); g[k:] = -1
    rows, _ = exact_rows(tri, g, pick=False, order=False)
    zero, one = fmpq(0), fmpq(1)
    e = lambda i: [one if c == i else zero for c in range(D)]  # noqa: E731
    rows = rows + [[a - b for a, b in zip(e(p1), e(p2))]] + [[a - b for a, b in zip(e(p3), e(l))] for l in I if l not in tri]
    for r in rows:
        assert all(r[c] == 0 for c in range(D) if c not in I)
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
    for a in rows:                                         # a.x >= 0 -> on the slice
        c = [a[j] for j in yidx]; c[ip3] += a[p2]
        if not add(c, -a[p2]):
            return (0.0, 0.0, {}) if return_info else (0.0, 0.0)
    for i in range(n):                                     # N on I: |m_j| <= NORM
        ee = [zero] * n; ee[i] = one
        add(ee, fmpq(-norm) - (1 if i == ip3 else 0)); add([-v for v in ee], fmpq(-norm) + (1 if i == ip3 else 0))
    A = [list(c) for c in out]; b = [out[c] for c in out]
    keep = irredundant(A, b)
    Af, bf = A, b
    A = [A[i] for i in keep]; b = [b[i] for i in keep]
    P = Poly(A, b); v0 = P.start()
    if v0 is None or v0 == "infeasible":
        raise RuntimeError("no start vertex")
    V, inc = P.enumerate(v0)
    drop = [i for i in range(len(Af)) if i not in set(keep)]
    if drop:
        Ad = fmpq_mat(len(drop), n, [v for i in drop for v in Af[i]])
        for v in V:
            s = Ad * fmpq_mat(n, 1, v)
            if any(s[kk, 0] < bf[i] for kk, i in enumerate(drop)):
                raise RuntimeError("dropped row violated")
    S, dim = pulling_triangulation(V, inc, n)
    if dim < n:
        return (0.0, 0.0, dict(dim=dim)) if return_info else (0.0, 0.0)

    def mvec(y):
        x = {j: y[q_] for q_, j in enumerate(yidx)}; x[p2] = 1 + x[p3]
        return [x[j] for j in I]
    M = [mvec(v) for v in V]
    t0q = fmpq(*float(t0).as_integer_ratio())
    zg = [t0q * zratio ** i for i in range(zbins)] + [None]
    c0 = (2 * arb.pi()) ** (-arb(d) / 2)

    def bounds(Ys, Ms, vol):
        if tangent:
            return _bounds_tangent(Ys, Ms, vol, d, zg, mlow, ip3, c0)
        smin = min_sq_lower(Ms); smax = max(sq(p) for p in Ms)
        yp_hi = max(y[ip3] for y in Ys); yp_lo = min(y[ip3] for y in Ys)
        hi = arb(0); lo = arb(0)
        for a, bb in zip(zg[:-1], zg[1:]):
            # sup / inf of Phi(z y_p3)^mlow over the simplex x [a, bb]
            if mlow:
                zt = (bb if yp_hi >= 0 else a)
                sup_phi = arb(1) if zt is None else (-(arb(zt) * arb(yp_hi)) / arb(2).sqrt()).erfc() / 2
                zl = (a if yp_lo >= 0 else bb)
                inf_phi = arb(0) if zl is None and yp_lo < 0 else (-(arb(zl) * arb(yp_lo)) / arb(2).sqrt()).erfc() / 2
                sup_phi, inf_phi = _ipow(sup_phi, mlow), _ipow(inf_phi, mlow)
            else:
                sup_phi = inf_phi = arb(1)
            hi += _G_arb(d, smin, a, bb) * sup_phi
            lo += _G_arb(d, smax, a, bb) * inf_phi
        return c0 * vol * lo, c0 * vol * hi

    import heapq as hq
    heap = []; tl = arb(0); th = arb(0); cnt = 0
    fact = math.factorial(n)
    for s in S:
        Ys = [V[i] for i in s]; Ms = [M[i] for i in s]
        vol = abs(fmpq_mat(n, n, [Ys[i][kk] - Ys[0][kk] for i in range(1, n + 1) for kk in range(n)]).det()) / fact
        lo, hi = bounds(Ys, Ms, vol); tl += lo; th += hi; cnt += 1
        hq.heappush(heap, (-float((hi - lo).upper()), cnt, Ys, Ms, vol, lo, hi))
    while heap and float(th.upper()) > tol * float(tl.lower()) and len(heap) < budget:
        _, _, Ys, Ms, vol, lo, hi = hq.heappop(heap); tl -= lo; th -= hi
        i, j = max(itertools.combinations(range(len(Ms)), 2), key=lambda e_: float(sq([a - b for a, b in zip(Ms[e_[0]], Ms[e_[1]])])))
        ym = [(a + b) / 2 for a, b in zip(Ys[i], Ys[j])]; mm = [(a + b) / 2 for a, b in zip(Ms[i], Ms[j])]
        for kk in (i, j):
            Yc = list(Ys); Mc = list(Ms); Yc[kk] = ym; Mc[kk] = mm
            lo2, hi2 = bounds(Yc, Mc, vol / 2); tl += lo2; th += hi2; cnt += 1
            hq.heappush(heap, (-float((hi2 - lo2).upper()), cnt, Yc, Mc, vol / 2, lo2, hi2))
    ub = float(np.nextafter(float(th.upper()), np.inf)); lb = max(0.0, float(np.nextafter(float(tl.lower()), -np.inf)))
    info = dict(d=d, mlow=mlow, vertices=len(V), simplices0=len(S), simplices=len(heap), tl=str(tl), th=str(th), heap=heap if return_info == "diag" else None)
    return (lb, ub, info) if return_info else (lb, ub)

"""T420: independent exact checks of a cell's geometry (V, inc, S, vlo, vhi) produced by geom_fast.

Shares no code with geom_fast / rigour beyond python-flint integer and rational types:
- vertices and incidences of {y : A y >= b} by cddlib in GMP rational arithmetic (pycddlib cdd.gmp);
- exact polytope volume by a Lasserre-type recursion over the face lattice built from cdd's incidences
  (apex = largest vertex index; facet heights by integer null vectors; coordinate projections keep it rational);
- exact simplex volumes by homogeneous integer determinants (rows [den_i, den_i * v_i]);
- ridge check: every ridge of S lies on a facet of Q and in one simplex, or is in two simplices on opposite sides.
Ridge check + exact volume equality prove that S is a triangulation of Q (covering degree is constant across
interior ridges and its integral equals vol(Q))."""
import math
from fractions import Fraction

import cdd.gmp as cg
import cdd
from flint import fmpq, fmpz, fmpz_mat


def fq(x):
    return Fraction(int(x.p), int(x.q))


def cdd_vertices(A, b):
    """Exact vertices and incidences (sets of row indices of A) of {y : A y >= b} by cddlib (GMP)."""
    rows = [[-fq(bi)] + [fq(a) for a in r] for r, bi in zip(A, b)]
    mat = cg.matrix_from_array(rows, rep_type=cdd.RepType.INEQUALITY)
    poly = cg.polyhedron_from_matrix(mat)
    gen = cg.copy_generators(poly)
    inc = cg.copy_incidence(poly)
    W, I = [], []
    for g, s in zip(gen.array, inc):
        if g[0] != 1 or gen.lin_set:
            raise ValueError("unbounded or has lineality")
        W.append(tuple(fmpq(x.numerator, x.denominator) for x in g[1:])); I.append(frozenset(s))
    return W, I


def homog(V):
    """Integer homogeneous rows [den, den * v] per vertex."""
    out = []
    for v in V:
        den = 1
        for x in v:
            den = den * int(x.q) // math.gcd(den, int(x.q))
        out.append([den] + [int(x.p) * (den // int(x.q)) for x in v])
    return out


def simplex_vols(Z, S, n, signs=None):
    """Exact volumes |det [den_i, z_i]| / prod(den_i) / n! for each simplex (index tuples into Z); if signs is a
    list, the sign of each det (rows in the order of s) is appended to it."""
    f = math.factorial(n)
    out = []
    for s in S:
        d = fmpz_mat(n + 1, n + 1, [x for i in s for x in Z[i]]).det()
        den = 1
        for i in s:
            den *= Z[i][0]
        out.append(fmpq(abs(int(d)), den * f))
        if signs is not None:
            signs.append((d > 0) - (d < 0))
    return out


def tree_sum(xs):
    """Exact sum by pairwise reduction (sequential summation of rationals with growing denominators is quadratic)."""
    xs = list(xs) or [fmpq(0)]
    while len(xs) > 1:
        xs = [xs[i] + xs[i + 1] if i + 1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
    return xs[0]


def _parity(seq):
    """+1 / -1: sign of the permutation sorting seq (distinct items)."""
    p = 1; a = list(seq)
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] > a[j]:
                p = -p
    return p


def lasserre_volume(W, I, n):
    """Exact n-volume of conv(W) (W the vertices, I their incidence sets) by recursion over facets:
    vol_C(F) = (1/d) sum_{G facet of F, a not in G} |h.Z_a| / (den_a |c_j|) vol_{C - j}(G), a = max(F),
    with vol_C the volume of the projection onto coordinates C (injective on aff F)."""
    Z = homog(W)
    m = 1 + max((max(s) for s in I if s), default=-1)
    rowmask = [0] * m
    for i, s in enumerate(I):
        for r in s:
            rowmask[r] |= 1 << i
    full = (1 << len(W)) - 1
    if fmpz_mat(len(W), n + 1, [x for z in Z for x in z]).rank() != n + 1:
        return fmpq(0)
    rank_memo, memo = {}, {}

    def idx(mask):
        out = []
        while mask:
            lb = mask & -mask; out.append(lb.bit_length() - 1); mask ^= lb
        return out

    def hrank(mask):                                     # affine rank + 1 (full coordinates)
        if mask not in rank_memo:
            ii = idx(mask)
            rank_memo[mask] = fmpz_mat(len(ii), n + 1, [x for i in ii for x in Z[i]]).rank()
        return rank_memo[mask]

    def vol(F, C, d):
        key = (F, C)
        if key in memo:
            return memo[key]
        ii = idx(F)
        cols = (0,) + tuple(c + 1 for c in C)
        if d == 0:
            res = fmpq(1)
        elif len(ii) == d + 1:
            det = fmpz_mat(d + 1, d + 1, [Z[i][c] for i in ii for c in cols]).det()
            den = 1
            for i in ii:
                den *= Z[i][0]
            res = fmpq(abs(int(det)), den * math.factorial(d))
        else:
            a = ii[-1]
            rows = set()
            for i in ii:
                rows |= I[i]
            facets = set()
            for r in rows:
                G = F & rowmask[r]
                if G != F and not (G >> a) & 1 and G not in facets and bin(G).count("1") >= d and hrank(G) == d:
                    facets.add(G)
            res = fmpq(0)
            for G in facets:
                gi = idx(G)
                H = fmpz_mat(len(gi), d + 1, [Z[i][c] for i in gi for c in cols])
                N, nul = H.nullspace()
                if nul != 1:
                    raise AssertionError("facet null space")
                h = [int(N[k, 0]) for k in range(d + 1)]
                j = next(k for k in range(1, d + 1) if h[k] != 0)
                ha = abs(sum(h[k] * Z[a][cols[k]] for k in range(d + 1)))
                Cg = C[:j - 1] + C[j:]
                res += fmpq(ha, Z[a][0] * abs(h[j])) * vol(G, Cg, d - 1)
            res = res / d
        memo[key] = res
        return res
    return vol(full, tuple(range(n)), n)


def ridge_check(S, signs, inc, n):
    """None if every ridge is a boundary ridge (common tight row) in one simplex or an interior ridge in two
    simplices with apexes strictly on opposite sides; else a description of the first failure. Side of apex a
    against ridge r: sign det[Z_r (sorted), Z_a]. For s sorted with sign det[Z_s] = sg, removing s[k] and appending it
    is n - k transpositions, so side = sg * (-1)^(n - k); an unsorted s first gets sg times the parity of its sort.
    Ridges are handled as a sorted numpy array (a dict of tuples does not fit 2.5 GB at ~2.6e6 ridges)."""
    import numpy as np
    S = np.asarray(S, dtype=np.int32); sg = np.asarray(signs, dtype=np.int8)
    inv = np.zeros(len(S), dtype=np.int64)
    for a in range(n + 1):
        for b in range(a + 1, n + 1):
            inv += S[:, a] > S[:, b]
    sg = sg * np.where(inv % 2 == 0, 1, -1).astype(np.int8)
    S = np.sort(S, axis=1)
    R = np.concatenate([np.delete(S, k, axis=1) for k in range(n + 1)])
    side = np.concatenate([sg * (1 if (n - k) % 2 == 0 else -1) for k in range(n + 1)]).astype(np.int8)
    del S
    o = np.lexsort(R.T[::-1]); R = R[o]; side = side[o]; del o
    new = np.r_[True, (R[1:] != R[:-1]).any(axis=1)]
    start = np.flatnonzero(new); size = np.diff(np.r_[start, len(R)])
    if (size > 2).any():
        return ("ridge in >2 simplices", tuple(int(x) for x in R[start[np.argmax(size > 2)]]))
    two = start[size == 2]
    bad = side[two] * side[two + 1] != -1
    if bad.any():
        return ("ridge apexes not on opposite sides", tuple(int(x) for x in R[two[np.argmax(bad)]]))
    masks = []
    for t in inc:
        m = 0
        for r in t:
            m |= 1 << r
        masks.append(m)
    for i in start[size == 1]:
        m = -1
        for v in R[i]:
            m &= masks[v]
        if m == 0:
            return ("interior ridge in one simplex", tuple(int(x) for x in R[i]))
    return None


def check_cell(A_full, b_full, A_kept, V, inc, S, vlo, vhi, n, ridge_max=300_000):
    """All independent checks; returns dict of results (ok flags and counts)."""
    out = {}
    # cdd on the kept rows; every dropped row is then checked exactly at every cdd vertex, so conv(W) satisfies
    # it and {A_kept y >= b_kept} = {A_full y >= b_full} (bounded); incidences are taken on the kept rows
    kept_set = {tuple(r) for r in A_kept}
    b_kept = [None] * len(A_kept); krow = {tuple(r): k for k, r in enumerate(A_kept)}
    for r, bi in zip(A_full, b_full):
        if tuple(r) in krow:
            b_kept[krow[tuple(r)]] = bi
    W, I = cdd_vertices(A_kept, b_kept)
    out["dropped_rows"] = sum(tuple(r) not in kept_set for r in A_full)
    out["dropped_ok"] = all(sum((a * w for a, w in zip(r, v)), fmpq(0)) >= bi
                            for r, bi in zip(A_full, b_full) if tuple(r) not in kept_set for v in W)
    pos = {tuple(v): j for j, v in enumerate(V)}
    out["cdd_nV"] = len(W)
    out["vset_ok"] = len(W) == len(V) and all(tuple(w) in pos for w in W)
    # incidences of V against the kept rows: exact slacks (fmpq) recomputed here, and cdd's (full rows) agree
    inc_ok = True
    if out["vset_ok"]:
        for w, s in zip(W, I):
            j = pos[tuple(w)]
            if s != inc[j]:
                inc_ok = False; break
    out["inc_ok"] = inc_ok and out["vset_ok"]
    Z = homog(V)
    signs = []
    out["nS"] = len(S)
    # streamed: each exact volume is checked against its enclosure and folded into a binary-counter pairwise sum
    # (storing 2.6e5 exact volumes and summing them pairwise does not fit 2.5 GB on the largest cells)
    enc = 0; nonpos = 0; stack = []
    for j in range(len(S)):
        v = simplex_vols(Z, [S[j]], n, signs)[0]
        if not v > 0:
            nonpos += 1
        if not (fmpq(*float(vlo[j]).as_integer_ratio()) <= v <= fmpq(*float(vhi[j]).as_integer_ratio())):
            enc += 1
        lev = 0
        while stack and stack[-1][0] == lev:
            v = stack.pop()[1] + v; lev += 1
        stack.append((lev, v))
    tot = fmpq(0)
    for _, v in reversed(stack):
        tot = tot + v
    del stack
    out["vol_pos_ok"] = nonpos == 0
    out["enclosure_fail"] = enc
    lv = lasserre_volume(W, I, n)
    out["vol_equal"] = tot == lv
    out["vol_float"] = float(tot)
    if len(S) <= ridge_max:
        rc = ridge_check([tuple(s) for s in S], signs, inc, n)
        out["ridge_ok"] = rc is None
        if rc is not None:
            out["ridge_fail"] = repr(rc)[:300]
    out["ok"] = (out["vset_ok"] and out["dropped_ok"] and out["inc_ok"] and out["vol_pos_ok"] and enc == 0 and out["vol_equal"]
                 and out.get("ridge_ok", True))
    return out

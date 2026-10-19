"""T412: fast rigorous M5t. A batched float replacement for rigour._bounds_tangent in which every rounding error is
bounded and rounded outward (derivations: T412_README.md, section "Error analysis").

Interface
  q2iv(x)                                     floats lo <= x <= hi for an exact rational x
  bounds_batch(Mlo, Mhi, vlo, vhi, P)         rigorous per-simplex [lo, hi] of the M5t quantity
                                              c0 vol E_simplex[ int_z (z-weighted integrand) ], same as _bounds_tangent
  Params(d, mlow, jp3, t0, zbins, zratio)     the cell constants (jp3 = index of p3 in the m-coordinates I)
  mass_cell_fast(tri, gates, k, t0, ...)      rigour.mass_cell(..., tangent=True) with these bounds and batched
                                              refinement

Number model. Every float is an exact rational. Simplex vertices are carried as float enclosures Mlo <= M <= Mhi
of exact vertices (the refinement midpoints stay exact in the real geometry; only their enclosures are floats).
+, -, *, /, sqrt are IEEE round-to-nearest, so nextafter of the computed result, outward, encloses the exact one
(subnormal results included). Sums of many terms use gamma_n = n u / (1 - n u), u = 2^-53, times SAFE = 2 (as
verify.py). exp, log, erfcx (libm / numpy / scipy) are ASSUMED within REL_T = 1e-13 relative (plus 4 * 2^-1074
absolute) and scipy.special.log_ndtr within LNDTR_ABS (1 + x^2) absolute; measured worst cases against arb are
1.2e-16, 7.2e-17, 6.7e-16 and 3.9e-16 (1 + x^2) (test_m5t_fast.py), i.e. >= 150x inside the assumed bounds."""
import itertools
import math
import os
import sys

import numpy as np
from flint import arb, fmpq, fmpq_mat
from numba import njit
from scipy.special import erfcx, log_ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rigour as R  # noqa: E402

U = 2.0 ** -53
SAFE = 2.0
REL_T = 1e-13
LNDTR_ABS = 1e-13
TINY = 4 * 2.0 ** -1074
WMAX = 5000.0           # exact E bound for knot spread w <= WMAX; above: knot cap (upper) / Jensen (lower)
EPS = 1e-8              # bins whose trivial upper term is <= EPS x the simplex total keep the trivial E bound
INF = np.inf
STATS = dict(exact_bins=0, simplices=0)


def gam(n):
    return n * U / (1 - n * U)


def up(x):
    """>= x + one ulp (see _up); applied to a just-rounded result it bounds the exact value from above."""
    return (x + np.abs(x) * 2.0 ** -52) + 2.0 ** -1074


def dn(x):
    return (x - np.abs(x) * 2.0 ** -52) - 2.0 ** -1074


def _q(f):
    return fmpq(*float(f).as_integer_ratio())


def q2iv(x):
    """Floats lo <= x <= hi for an exact rational x (checked exactly, no assumption on float(fmpq))."""
    x = fmpq(x); f = float(x.p) / float(x.q) if abs(x.p) < 2 ** 1000 and x.q < 2 ** 1000 else float(x)
    lo = hi = f
    while _q(lo) > x:
        lo = float(np.nextafter(lo, -np.inf))
    while _q(hi) < x:
        hi = float(np.nextafter(hi, np.inf))
    assert _q(lo) <= x <= _q(hi)
    return lo, hi


def _arbq(x):
    """Exact rational value of an exact arb (a ball endpoint)."""
    m, e = x.man_exp(); m, e = int(m), int(e)
    return fmpq(m * 2 ** e) if e >= 0 else fmpq(m, 2 ** -e)


def a2iv(a):
    """Floats enclosing an arb ball."""
    return q2iv(_arbq(a.lower()))[0], q2iv(_arbq(a.upper()))[1]


# ---------------------------------------------------------------- outward-rounded elementary pieces
def mul_pos_hi(alo, ahi, y):
    """Upper end of [alo, ahi] * y for alo >= 0 (exact float y)."""
    return up(np.where(y >= 0, ahi, alo) * y)


def mul_pos_lo(alo, ahi, y):
    return dn(np.where(y >= 0, alo, ahi) * y)


def exp_up(x):
    return up(up(np.exp(x) * (1 + 2 * REL_T)) + TINY)


def exp_dn(x):
    return np.maximum(0.0, dn(dn(np.exp(x) * (1 - 2 * REL_T)) - TINY))


def lndtr_iv(xlo, xhi):
    """Enclosure of log Phi(x) for x in [xlo, xhi] (log Phi increasing)."""
    lo = log_ndtr(xlo); hi = lo if xhi is xlo else log_ndtr(xhi)
    return dn(lo - up(LNDTR_ABS * SAFE * (1 + xlo * xlo))), up(hi + up(LNDTR_ABS * SAFE * (1 + xhi * xhi)))


def idot(c, lo, hi):
    """Enclosure of sum_k c_k x_k (exact float c, x_k in [lo_k, hi_k]) along the last axis."""
    mid = 0.5 * lo + 0.5 * hi
    rad = np.maximum(up(hi - mid), up(mid - lo))
    n = lo.shape[-1]
    v = (c * mid).sum(-1)
    err = up(SAFE * ((np.abs(c) * rad).sum(-1) + gam(n + 2) * (np.abs(c * mid).sum(-1) + (np.abs(c) * rad).sum(-1))))
    return dn(v - err), up(v + err)


def sumsq_up(a):
    """Upper bound on sum a_k^2 along the last axis (a exact floats)."""
    return up((a * a).sum(-1) * (1 + SAFE * gam(a.shape[-1] + 1)))


def sumsq_dn(a):
    return dn((a * a).sum(-1) * (1 - SAFE * gam(a.shape[-1] + 1)))


# ---------------------------------------------------------------- E_exp: mean of exp(-u) over a simplex
WSER = 24.0             # series for w <= WSER, scaling and squaring above


@njit(cache=True, inline="always")
def _up(x):
    """>= x + one ulp of x (or the next subnormal): |x| 2^-52 is in [ulp(x), 2 ulp(x)) for normal x, so
    x + |x| 2^-52 rounds to at least the next float; + 2^-1074 covers 0 and subnormals. Used on a just-rounded
    result, it bounds the exact value from above (error <= ulp / 2)."""
    return (x + abs(x) * 2.0 ** -52) + 2.0 ** -1074


@njit(cache=True, inline="always")
def _dn(x):
    return (x - abs(x) * 2.0 ** -52) - 2.0 ** -1074


@njit(cache=True)
def _e1(t, upper, rel, wcut, s, a, b, cap=False):
    """t: vertex values t_0..t_n of an affine u (exact floats: for an upper bound the lower ends of the true knot
    enclosures, for a lower bound the upper ends; E is decreasing in every knot). Returns (T, LF, w) with
    E = E_unif exp(-u) <= exp(-T + LF) (upper) or >= exp(-T + LF) (lower). E = e^-T F(s), s_i = T - t_i >= 0,
    F(s) = n! e[s_0..s_n] = n! sum_k h_k(s) / (n + k)! (Hermite-Genocchi + Dirichlet moments; all terms >= 0).
    w > wcut: the cheap bounds E <= mean_i exp(-t_i) (convexity) / E >= exp(-mean t) (Jensen), or with cap (upper only) the knots above tmin + WMAX are
    lowered to it (valid: E decreases in every knot). s, a, b: work arrays of length n + 1."""
    V = t.shape[0]
    n = V - 1
    Tm = t[0]
    tmn = t[0]
    for i in range(1, V):
        if t[i] > Tm:
            Tm = t[i]
        if t[i] < tmn:
            tmn = t[i]
    if cap and upper and Tm - tmn > WMAX:
        Tm = tmn + WMAX                         # any float; knots min(t_i, Tm) <= t_i
        wcut = 2 * WMAX
    w = 0.0
    for i in range(V):
        d = Tm - min(t[i], Tm)                  # a float difference is 0 only if exact
        if d != 0.0:
            d = _up(d) if upper else max(0.0, _dn(d))
        s[i] = d
        if d > w:
            w = d
    if n == 0 or w == 0.0:
        return Tm, 0.0, w                       # E = e^-T exactly (s computed exactly 0)
    if w > wcut and wcut < 0.0:                 # trivial bounds e^-tmin / e^-tmax
        return Tm, (w if upper else 0.0), w
    if w > wcut:                                # cheap bounds: convexity / Jensen
        if upper:                               # e^-u <= sum_i lambda_i e^-t_i: F <= mean_i e^(s_i) = e^w mean e^(s_i - w)
            acc = 0.0
            for i in range(V):
                acc = _up(acc + _up(np.exp(_up(s[i] - w)) * (1 + 2 * rel)))
            acc = _up(acc / V)
            return Tm, min(w, _up(w + _up(np.log(acc) + 2 * rel * abs(np.log(acc))))), w
        acc = 0.0                               # E >= e^-(mean t): F >= e^(mean s)
        for i in range(V):
            acc = _dn(acc + s[i])
        return Tm, max(0.0, _dn(acc / V)), w
    if w > WSER:
        return Tm, _emat(s, n, upper, rel), w
    u = 2.0 ** -53
    for j in range(V):
        a[j] = 1.0                              # H_0^(j) = c_0 h_0 = 1
    S = 1.0
    tk = 1.0                                    # >= w^k / k!
    k = 0
    t1 = 1.0
    C = 1.0 + 2.0 ** -50                        # fl(fl(fl(x w) / k) C) >= x w / k for x, w > 0 (relative error <= 3u)
    while True:
        k += 1
        inv = 1.0 / (n + k)
        prev = 0.0
        for j in range(V):                      # H_k^(j) = H_k^(j-1) + s_j H_{k-1}^(j) / (n + k), H_k = n! h_k/(n+k)!
            prev = prev + s[j] * a[j] * inv
            b[j] = prev
        S = S + prev
        for j in range(V):
            a[j] = b[j]
        tk = tk * w / k * C
        t1 = tk * w / (k + 1) * C
        if k + 2 >= 2.0 * w and 2.0 * t1 <= 2.0 ** -60 * S:
            break
    D = 6 * k + n + 2
    g = D * u / (1 - D * u)
    if upper:
        F = _up(S * (1 + 2 * g) + 4.0 * t1 + 1e-50)
        return Tm, min(_up(np.log(F) * (1 + 2 * rel)), w), w
    F = S * (1 - 2 * g) - 1e-50
    return Tm, (max(0.0, _dn(np.log(F) * (1 - 2 * rel))) if F > 1.0 else 0.0), w


@njit(cache=True)
def _eser(K, upper, rel, wcut):
    M, V = K.shape
    T = np.empty(M)
    LF = np.empty(M)
    s = np.empty(V); a = np.empty(V); b = np.empty(V)
    for r in range(M):
        T[r], LF[r], _ = _e1(K[r], upper, rel, wcut, s, a, b)
    return T, LF


@njit(cache=True)
def _emat(s, n, upper, rel):
    """log of a bound on F = n! e[s_0..s_n] (divided difference of exp, s_i >= 0) by scaling and squaring:
    e[s_i..s_l] = exp(Z)_il for the upper bidiagonal Z (diag s, superdiagonal 1) (Opitz), exp(Z) = exp(Z/2^m)^(2^m).
    exp(Z/2^m)_il = 2^(-m(l-i)) e[x_i..x_l], x = s/2^m in [0, 1], from the all-positive series
    e[x_i..x_l] = sum_k h_k(x_i..x_l) / (l-i+k)!. Every quantity is >= 0, so round-to-nearest errors are bounded by
    gamma_D of the longest rounding chain D (T412_README.md)."""
    V = n + 1
    w = 0.0
    for i in range(V):
        if s[i] > w:
            w = s[i]
    m = 0
    while 2.0 ** m < w:
        m += 1                                  # x = s / 2^m in [0, 1]
    sc = 2.0 ** -m
    x = np.empty(V)
    for i in range(V):
        si = s[i]
        if upper:
            si = max(si, 2.0 ** -500)           # raising s raises F (upper); keeps every product normal
        elif si < 2.0 ** -500:
            si = 0.0
        x[i] = si * sc
    K = 18                                      # relative tail <= 1.1 / 19! < 1e-17
    c = np.empty(K + V + 1)                     # c_j ~ 1/j! (j roundings)
    c[0] = 1.0
    for j in range(1, K + V + 1):
        c[j] = c[j - 1] / j
    A = np.empty(K + 1)
    M = np.zeros((V, V))
    u = 2.0 ** -53
    DT = 6 * K + 3 * V + 6
    gT = DT * u / (1 - DT * u)
    for i in range(V):
        A[0] = 1.0
        for k in range(1, K + 1):
            A[k] = 0.0
        for l in range(i, V):
            xl = x[l]
            for k in range(1, K + 1):           # h_k(x_i..x_l) = h_k(x_i..x_{l-1}) + x_l h_{k-1}(x_i..x_l)
                A[k] = A[k] + xl * A[k - 1]
            acc = 0.0
            for k in range(K + 1):
                acc += A[k] * c[l - i + k]
            if upper:
                acc = _up(acc * (1 + 2 * gT) + 1.2 * c[K + 1] * c[l - i])   # tail <= 1.1 / ((K+1)! (l-i)!)
            else:
                acc = max(0.0, _dn(acc * (1 - 2 * gT)))
            M[i, l] = acc * 2.0 ** (-m * (l - i))
    xm = 0.0
    for i in range(V):
        xm = max(xm, x[i])
    f = _exp_up(-xm, rel) if upper else _exp_dn(-xm, rel)   # exp(X - xm I) = e^-xm exp(X): entries <= 1, no overflow
    for i in range(V):
        for l in range(i, V):
            M[i, l] = _up(M[i, l] * f) if upper else _dn(M[i, l] * f)
    T = np.zeros((V, V))
    for _ in range(m):                          # m squarings
        for i in range(V):
            for l in range(i, V):
                acc = 0.0
                for k in range(i, l + 1):
                    acc += M[i, k] * M[k, l]
                T[i, l] = acc
        for i in range(V):
            for l in range(i, V):
                M[i, l] = T[i, l]
    fact = 1.0
    for i in range(2, n + 1):
        fact *= i
    D = (2.0 ** m) * (n + 1) + 2
    g = D * u / (1 - D * u)
    sm = xm * 2.0 ** m                          # exact: exp(Z) = e^sm exp(Z - sm I)
    if upper:                                   # + underflow slack: (2V+2)^(m+1) (K+V) 2^-1074 (T412_README.md)
        F = _up(M[0, n] * fact * (1 + 2 * g) + fact * (2.0 * V + 2.0) ** (m + 1) * (K + V) * 2.0 ** -1074)
        lg = np.log(F)
        return min(_up(sm + _up(lg + 2 * rel * abs(lg))), w)
    F = M[0, n] * fact * (1 - 2 * g)
    if F <= 0.0:
        return 0.0
    lg = np.log(F)
    return max(0.0, _dn(sm + _dn(lg - 2 * rel * abs(lg))))


def E_exp(K, upper, wcut=WMAX):
    """Batched _e1 over the last axis of K: (T, LF)."""
    sh = K.shape
    T, LF = _eser(np.ascontiguousarray(K.reshape(-1, sh[-1])), upper, REL_T, min(wcut, WMAX))
    return T.reshape(sh[:-1]), LF.reshape(sh[:-1])


@njit(cache=True, inline="always")
def _exp_up(x, rel):
    return _up(_up(np.exp(x) * (1 + 2 * rel)) + 4 * 2.0 ** -1074)


@njit(cache=True, inline="always")
def _exp_dn(x, rel):
    return max(0.0, _dn(_dn(np.exp(x) * (1 - 2 * rel)) - 4 * 2.0 ** -1074))


@njit(cache=True)
def _bin_terms(i, bb, tshlo, tshhi, yplo, yphi, llo, lhi, mlow, kap, kal, Lu, Ll, Gu, Gl, rel, wcut, kt, s, a, c, cap):
    """Upper / lower bound of the bin term (simplex i, finite z-bin bb) and the largest knot spread."""
    V = tshlo.shape[1]
    for v in range(V):
        kt[0, v] = max(0.0, _dn(llo[bb] * tshlo[i, v]))         # llo tsh, lower ends (upper bound, const Phi^m)
        kt[1, v] = _up(lhi[bb] * tshhi[i, v])                   # lhi tsh, upper ends (lower bound)
    if mlow > 0:
        for v in range(V):                                      # - kappa y_p3: kappa >= 0, sign of y decides the end
            y = yphi[i, v]
            kt[2, v] = _dn(kt[0, v] - _up((kap[1, i, bb] if y >= 0 else kap[0, i, bb]) * y))
            y = yplo[i, v]
            kt[3, v] = _up(kt[1, v] - _dn((kal[0, i, bb] if y >= 0 else kal[1, i, bb]) * y))
    nt = 4 if mlow > 0 else 2
    hu = np.inf
    hl = 0.0
    wm = 0.0
    if wcut < 0.0:                              # trivial: E <= exp(-min knot) / E >= exp(-max knot)
        for j in range(nt):
            mn = kt[j, 0]; mx = kt[j, 0]
            for v in range(1, V):
                mn = min(mn, kt[j, v]); mx = max(mx, kt[j, v])
            L = j // 2
            if j % 2 == 0:
                hu = min(hu, _up(_exp_up(_up(Lu[L, i, bb] - mn), rel) * Gu[i, bb]))
            else:
                hl = max(hl, _dn(_exp_dn(_dn(Ll[L, i, bb] - mx), rel) * Gl[i, bb]))
            wm = max(wm, mx - mn)
        return hu, hl, wm
    for jj in range(nt):
        j = (jj + 2) % nt                       # tangent / chord first; the constant-Phi^m variant only if it can win
        upper = j % 2 == 0
        if j < 2 and nt == 4:
            mn = kt[j, 0]; mx = kt[j, 0]
            for v in range(1, V):
                mn = min(mn, kt[j, v]); mx = max(mx, kt[j, v])
            if upper and _exp_dn(_dn(Lu[0, i, bb] - mx), rel) * Gu[i, bb] >= hu:   # its value >= this >= hu
                continue
            if not upper and _exp_up(_up(Ll[0, i, bb] - mn), rel) * Gl[i, bb] <= hl:
                continue
        T, LF, w = _e1(kt[j], upper, rel, wcut, s, a, c, cap)
        wm = max(wm, w)
        L = j // 2
        if upper:
            hu = min(hu, _up(_exp_up(_up(_up(Lu[L, i, bb] - T) + LF), rel) * Gu[i, bb]))
        else:
            hl = max(hl, _dn(_exp_dn(_dn(_dn(Ll[L, i, bb] - T) + LF), rel) * Gl[i, bb]))
    return hu, hl, wm


@njit(cache=True)
def _bins(tshlo, tshhi, yplo, yphi, llo, lhi, mlow, kap, kal, Lu, Ll, Gu, Gl, rel, wmax, eps):
    """Sum over the finite z-bins of the bin terms, per simplex, rounded outward. All bins first get the trivial
    E bound (exp(-min knot) / exp(-max knot)); then, largest upper term first, bins get the series / matrix E bound
    until the largest remaining (trivial) upper term is <= eps x the current total. Every bin term is a valid
    bound either way; the order only decides tightness."""
    N, V = tshlo.shape
    nb = llo.shape[0]
    hs = np.empty(N)
    ls = np.empty(N)
    nex = np.zeros(N, np.int64)
    kt = np.empty((4, V))
    s = np.empty(V); a = np.empty(V); c = np.empty(V)
    hu = np.empty(nb); hl = np.empty(nb); wm = np.empty(nb)
    for i in range(N):
        for bb in range(nb):                    # trivial E bounds everywhere (wcut < 0)
            hu[bb], hl[bb], wm[bb] = _bin_terms(i, bb, tshlo, tshhi, yplo, yphi, llo, lhi, mlow, kap, kal, Lu, Ll, Gu, Gl, rel, -1.0, kt, s, a, c, False)
        for bb in np.argsort(-hu):              # exact E for the largest terms; stop once the rest is negligible
            tot = 0.0
            for q_ in range(nb):
                tot += hu[q_]
            if np.isfinite(tot) and hu[bb] <= eps * tot:
                break
            hu[bb], hl[bb], _ = _bin_terms(i, bb, tshlo, tshhi, yplo, yphi, llo, lhi, mlow, kap, kal, Lu, Ll, Gu, Gl, rel, wmax, kt, s, a, c, True)
            nex[i] += 1
        h = 0.0
        l = 0.0
        for bb in range(nb):
            h = _up(h + hu[bb])
            l = _dn(l + hl[bb])
        hs[i] = h
        ls[i] = max(0.0, l)
    return hs, ls, nex


# ---------------------------------------------------------------- z-integral G(d, s, a, b) = e^-xa * Gm
class Params:
    """Cell constants: z-bin edges (exact floats), c0 = (2 pi)^(-d/2) and log sqrt(2 pi) enclosures, the
    incomplete-gamma coefficients for h = d/2."""
    def __init__(self, d, mlow, jp3, t0, zbins=24, zratio=fmpq(23, 20)):
        self.d, self.mlow, self.jp3 = d, mlow, jp3
        t0q = _q(t0)
        self.edges = np.array([float(t0q * zratio ** i) for i in range(zbins)])   # exact floats; e_0 = t0
        assert self.edges[0] == t0 and np.all(np.diff(self.edges) > 0)
        self.c0 = a2iv((2 * arb.pi()) ** (-arb(d) / 2))
        self.ls2p = a2iv((2 * arb.pi()).sqrt().log())
        self.half = d % 2 == 1
        hm = d // 2                                                                  # h = d/2 = hm (+ 1/2)
        if self.half:
            self.gh = a2iv(arb(fmpq(d, 2)).gamma())
            self.gk = [float(np.prod([j + 0.5 for j in range(k, hm)])) for k in range(1, hm + 1)]  # G(h)/G(k+1/2)
        else:
            self.pc = [float(math.factorial(hm - 1) // math.factorial(j)) for j in range(hm)]  # (h-1)!/j!
        self.hm = hm
        self.coef = np.array(self.gk if self.half else self.pc)
        self.ghv = np.array(self.gh if self.half else (1.0, 1.0))


def _P(P, xlo, xhi, upper):
    """Bound on P(x) = e^x Gamma(h, x) for x in [xlo, xhi] (each term bounded on its own)."""
    rd = up if upper else dn
    x = xhi if upper else xlo
    if not P.half:
        v = np.full_like(x, P.pc[-1])
        for c in P.pc[-2::-1]:
            v = rd(rd(v * x) + c)
        return v
    sx = rd(np.sqrt(x))
    sl = dn(np.sqrt(xlo)) if upper else up(np.sqrt(xhi))                           # erfcx decreasing
    e = erfcx(sl)
    v = rd(rd((P.gh[1] if upper else P.gh[0]) * e) * ((1 + 2 * REL_T) if upper else (1 - 2 * REL_T)))
    pw = sx
    for k in range(1, P.hm + 1):                                                    # x^(k-1/2)
        v = rd(v + rd(P.gk[k - 1] * pw))
        pw = rd(pw * x)
    return v


def _C(P, s, upper):
    """(2/s)^h / 2 for exact float s > 0."""
    rd = up if upper else dn
    q = rd(2.0 / s); v = np.ones_like(q)
    for _ in range(P.hm):
        v = rd(v * q)
    if P.half:
        v = rd(v * rd(np.sqrt(q)))
    return 0.5 * v


def G_fac(P, s, A, B, upper):
    """(Lexp, Gm): G(d, s, A, B) <= exp(Lexp) Gm (upper) or >= (lower), s exact float, B = inf allowed."""
    rd = up if upper else dn
    AA = (dn(A * A), up(A * A))
    xa = (dn(dn(AA[0] * s) * 0.5), up(up(AA[1] * s) * 0.5))
    C = _C(P, s, upper)
    if np.isinf(B).all():
        Gm = rd(C * _P(P, xa[0], xa[1], upper))
    else:
        BB = (dn(B * B), up(B * B))
        xb = (dn(dn(BB[0] * s) * 0.5), up(up(BB[1] * s) * 0.5))
        dl = (dn(dn(dn(BB[0] - AA[1]) * s) * 0.5), up(up(up(BB[1] - AA[0]) * s) * 0.5))
        if upper:
            Gm = up(C * up(_P(P, *xa, True) - dn(exp_dn(-dl[1]) * _P(P, *xb, False))))
        else:
            Gm = dn(C * np.maximum(0.0, dn(_P(P, *xa, False) - up(exp_up(-dl[0]) * _P(P, *xb, True)))))
    return (-xa[0] if upper else -xa[1]), Gm


# ---------------------------------------------------------------- per-simplex bounds
@njit(cache=True)
def _min_norm(Mf, iters):
    """Barycentric weights of the min-norm point of conv(rows of Mf[i]) by Wolfe's algorithm (float). Only a
    proposal: m_c enters the bounds as an exact float point, so its accuracy affects tightness only (rigour.py
    uses NNLS for the same proposal)."""
    N, V, d = Mf.shape
    out = np.zeros((N, V))
    G = np.empty((V, V))
    for i in range(N):
        P = Mf[i]
        for p in range(V):
            for q_ in range(V):
                acc = 0.0
                for k in range(d):
                    acc += P[p, k] * P[q_, k]
                G[p, q_] = acc
        scale = 0.0
        j0 = 0
        for p in range(V):
            scale = max(scale, G[p, p])
            if G[p, p] < G[j0, j0]:
                j0 = p
        inS = np.zeros(V, np.bool_)
        lam = np.zeros(V)
        inS[j0] = True; lam[j0] = 1.0
        for _ in range(iters):
            gx = G @ lam                         # P_j . x
            xx = lam @ gx
            j = int(np.argmin(gx))
            if xx - gx[j] <= 1e-14 * scale or inS[j]:
                break
            inS[j] = True
            for _m in range(V + 2):              # minor cycle
                idx = np.nonzero(inS)[0]
                k = idx.shape[0]
                A = np.zeros((k + 1, k + 1)); r = np.zeros(k + 1); r[k] = 1.0
                for a_ in range(k):
                    for b_ in range(k):
                        A[a_, b_] = G[idx[a_], idx[b_]]
                    A[a_, k] = 1.0; A[k, a_] = 1.0
                ok = True
                al = np.zeros(k)
                try:
                    sol = np.linalg.solve(A, r)
                    al = sol[:k]
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok or not np.all(np.isfinite(al)):
                    break
                if np.all(al > 1e-15):
                    for a_ in range(k):
                        lam[idx[a_]] = al[a_]
                    break
                th = 1.0
                for a_ in range(k):
                    if al[a_] <= 1e-15:
                        li = lam[idx[a_]]
                        th = min(th, li / (li - al[a_]) if li - al[a_] > 0 else 0.0)
                for a_ in range(k):
                    lam[idx[a_]] = (1 - th) * lam[idx[a_]] + th * al[a_]
                for a_ in range(k):
                    if lam[idx[a_]] <= 1e-15:
                        lam[idx[a_]] = 0.0; inS[idx[a_]] = False
                sl = lam.sum()
                lam /= sl
        out[i] = lam / lam.sum()
    return out


def min_norm_weights(Mf, iters=100):
    return _min_norm(np.ascontiguousarray(Mf), iters)


def simplex_consts(Mlo, Mhi, jp3, mc=None):
    """Per simplex: tangent centre m_c (float, exact by definition), enclosures of s_c, tau_i, R^2, s_min, s_max,
    y_p3 at the vertices. Mlo, Mhi: (N, V, d)."""
    N, V, d = Mlo.shape
    Mf = 0.5 * Mlo + 0.5 * Mhi
    if mc is None:                                                     # any float point is valid; this is rigour's
        mc = np.einsum("nv,nvd->nd", min_norm_weights(Mf), Mf)
    sc = (sumsq_dn(mc), sumsq_up(mc))
    dlo, dhi = idot(mc[:, None, :], Mlo, Mhi)                          # m_c . M_i
    tau = (2 * dn(dlo - sc[1][:, None]), 2 * up(dhi - sc[0][:, None]))                   # 2 m_c.(M_i - m_c)
    ab = np.maximum(np.abs(dn(Mlo - mc[:, None])), np.abs(up(Mhi - mc[:, None])))
    R2 = sumsq_up(ab).max(1)
    smax = sumsq_up(np.maximum(np.abs(Mlo), np.abs(Mhi))).max(1)
    L = dlo.min(1)                                                      # separating hyperplane through m_c
    smin = np.where(L > 0, dn(dn(L * L) / sc[1]), 0.0)
    smin = np.maximum(smin, 0.5)                                        # |m|^2 >= 1/2 on the slice
    return dict(mc=mc, sc=sc, tau=tau, R2=R2, smin=smin, smax=smax, yplo=Mlo[:, :, jp3], yphi=Mhi[:, :, jp3],
                ypc=mc[:, jp3])


CHUNK = 4096


def bounds_batch(Mlo, Mhi, vlo, vhi, P, mc=None):
    """Rigorous [lo, hi] per simplex (arrays (N,)). The quantity is the one of rigour._bounds_tangent. mc: optional
    tangent centres (N, d) (any floats; default: min-norm point of the simplex, as rigour). Chunked (cache)."""
    N = len(vlo)
    if N <= CHUNK:
        return _bounds_batch(Mlo, Mhi, vlo, vhi, P, mc)
    lo = np.empty(N); hi = np.empty(N)
    for a in range(0, N, CHUNK):
        b = min(N, a + CHUNK)
        lo[a:b], hi[a:b] = _bounds_batch(Mlo[a:b], Mhi[a:b], vlo[a:b], vhi[a:b], P, None if mc is None else mc[a:b])
    return lo, hi


def _bounds_batch_np(Mlo, Mhi, vlo, vhi, P, mc=None):
    """Numpy version of the per-(simplex, bin) set-up (reference for the numba port _prep1/_prep2)."""
    Mlo = np.asarray(Mlo, float); Mhi = np.asarray(Mhi, float)
    N, V, d = Mlo.shape
    c = simplex_consts(Mlo, Mhi, P.jp3, mc)
    mlow = P.mlow
    A = P.edges[None, :-1]; B = P.edges[None, 1:]                       # finite bins (1, nb)
    llo = (0.5 * dn(A * A), 0.5 * up(A * A)); lhi = (0.5 * dn(B * B), 0.5 * up(B * B))
    T0 = c["tau"][0].min(1)[:, None]                                    # split point (exact float) <= tau_i
    tshlo = np.maximum(0.0, dn(c["tau"][0] - T0))[:, None, :]          # (N, 1, V)
    tshhi = up(c["tau"][1] - T0)[:, None, :]
    E1u = np.where(T0 < 0, up(lhi[1] * (-T0)), up(-(llo[0] * T0)))     # >= -lambda T0, lambda in [llo, lhi]
    E1l = np.where(T0 < 0, dn(llo[0] * (-T0)), dn(-(lhi[1] * T0)))
    yplo = c["yplo"]; yphi = c["yphi"]
    ylo = c["yplo"].min(1)[:, None]; yhi = c["yphi"].max(1)[:, None]
    nb = A.shape[1]
    kap = np.zeros((2, N, nb)); kal = np.zeros((2, N, nb))
    Gu_e, Gu = G_fac(P, c["sc"][0][:, None], A, B, True)
    s_l = up(c["sc"][1] + c["R2"])[:, None]                            # s <= s_c + tau + R^2
    Gl_e, Gl = G_fac(P, s_l, A, B, False)
    Lu = np.empty((2, N, nb)); Ll = np.empty((2, N, nb))
    if mlow:
        neg = np.maximum(0.0, -ylo)
        BmA = up(B - A)
        x0 = (0.5 * (A + B)) * c["ypc"][:, None]                        # tangent point (any float)
        xl = np.minimum(dn(A * ylo), dn(B * ylo)); xh = np.maximum(up(A * yhi), up(B * yhi))
        lp0 = lndtr_iv(x0, x0)
        ex_lo = dn(dn(dn(-0.5 * up(x0 * x0)) - P.ls2p[1]) - lp0[1])     # log(phi/Phi)(x0)
        ex_hi = up(up(up(-0.5 * dn(x0 * x0)) - P.ls2p[0]) - lp0[0])
        rl, rh = exp_dn(ex_lo), exp_up(ex_hi)
        rt = np.exp(0.5 * ex_lo + 0.5 * ex_hi)                          # slope used (any float >= 0)
        rt = np.where(rt < 1e-200, 0.0, rt)
        Wx = np.maximum(up(xh - x0), up(x0 - xl))
        dlt = up(np.maximum(up(rh - rt), up(rt - rl)) * Wx)              # slope-error correction
        cu = up(lp0[1] + dlt); cu = up(cu + up(-(rt * x0))); cu = up(cu + up(up(rt * BmA) * neg)); cu = up(mlow * cu)
        kap[0] = np.maximum(0.0, dn(dn(mlow * rt) * B)); kap[1] = up(up(mlow * rt) * B)
        L1 = lndtr_iv(xl, xl)[0]; L2h = lndtr_iv(xh, xh)
        L2 = L2h[0]
        den = (dn(xh - xl), up(xh - xl)); num = (dn(L2 - L1), up(L2 - L1))
        ok = den[0] > 0
        dsafe = (np.where(ok, den[0], 1.0), np.where(ok, den[1], 1.0))
        bl = np.where(num[0] >= 0, dn(num[0] / dsafe[1]), dn(num[0] / dsafe[0]))
        bh = np.where(num[1] >= 0, up(num[1] / dsafe[0]), up(num[1] / dsafe[1]))
        bt = np.where(ok & (bl > 1e-200), bl, 0.0)                       # chord slope used (any float >= 0)
        eb = np.where(ok, np.maximum(0.0, np.maximum(up(bh - bt), up(bt - bl))), 0.0)
        cl = dn(L1 + dn(-(bt * xl))); cl = dn(cl - up(eb * den[1])); cl = dn(cl - up(up(bt * BmA) * neg)); cl = dn(mlow * cl)
        kal[0] = np.maximum(0.0, dn(dn(mlow * bt) * A)); kal[1] = up(up(mlow * bt) * A)
        lsup = up(mlow * L2h[1]); linf = dn(mlow * L1)
        Lu[0] = up(up(E1u + Gu_e) + lsup); Lu[1] = up(up(E1u + cu) + Gu_e)
        Ll[0] = dn(dn(E1l + Gl_e) + linf); Ll[1] = dn(dn(E1l + cl) + Gl_e)
    else:
        Lu[0] = up(E1u + Gu_e); Ll[0] = dn(E1l + Gl_e)
        Lu[1] = INF; Ll[1] = -INF
    tsl = np.ascontiguousarray(tshlo[:, 0, :]); tsh = np.ascontiguousarray(tshhi[:, 0, :])
    hsum, lsum, nex = _bins(tsl, tsh, np.ascontiguousarray(yplo), np.ascontiguousarray(yphi), np.ascontiguousarray(llo[0][0]),
                       np.ascontiguousarray(lhi[1][0]), mlow, kap, kal, Lu, Ll, np.ascontiguousarray(Gu),
                       np.ascontiguousarray(Gl), REL_T, WMAX, EPS)
    # last bin [A, inf)
    Al = np.full((N, 1), P.edges[-1]); Bi = np.full((N, 1), INF)
    e1, g1 = G_fac(P, c["smin"][:, None], Al, Bi, True)
    e2, g2 = G_fac(P, c["smax"][:, None], Al, Bi, False)
    if mlow:
        xs = up(Al * yhi)
        lsupi = np.where(yhi >= 0, 0.0, up(mlow * lndtr_iv(xs, xs)[1]))
        hinf = up(exp_up(up(e1 + lsupi)) * g1)
        xa = dn(Al * ylo)
        li = dn(mlow * lndtr_iv(xa, xa)[0])
        linf_ = np.where(ylo < 0, 0.0, dn(exp_dn(dn(e2 + li)) * g2))
    else:
        hinf = up(exp_up(e1) * g1); linf_ = dn(exp_dn(e2) * g2)
    hs = up(hsum + hinf[:, 0])
    ls = np.maximum(0.0, dn(lsum + linf_[:, 0]))
    hi = up(up(P.c0[1] * np.asarray(vhi)) * hs)
    lo = dn(dn(P.c0[0] * np.asarray(vlo)) * ls)
    STATS["exact_bins"] += int(nex.sum()); STATS["simplices"] += N
    return lo, hi


# ---------------------------------------------------------------- M5t cell bound with the fast per-simplex bounds
def cell_geometry(tri, gates, k, norm=R.NORM, event=False):
    """Exact slice polytope and pulling triangulation of rigour.mass_cell (copied; rigour.py is not edited).
    Returns None if empty / lower-dimensional, else (d, mlow, jp3, Mlo, Mhi, S, vlo, vhi)."""
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
    V, inc = Pl.enumerate(v0)
    drop = [i for i in range(len(Af)) if i not in set(keep)]
    if drop:
        Ad = fmpq_mat(len(drop), n, [v for i in drop for v in Af[i]])
        for v in V:
            s = Ad * fmpq_mat(n, 1, v)
            if any(s[kk, 0] < bf[i] for kk, i in enumerate(drop)):
                raise RuntimeError("dropped row violated")
    S, dim = R.pulling_triangulation(V, inc, n)
    if dim < n:
        return None

    def mvec(y):
        x = {j: y[q_] for q_, j in enumerate(yidx)}; x[p2] = 1 + x[p3]
        return [x[j] for j in I]
    iv = [[q2iv(x) for x in mvec(v)] for v in V]
    Mlo = np.array([[x[0] for x in r] for r in iv]); Mhi = np.array([[x[1] for x in r] for r in iv])
    fact = math.factorial(n)
    vols = [q2iv(abs(fmpq_mat(n, n, [V[s[i]][kk] - V[s[0]][kk] for i in range(1, n + 1) for kk in range(n)]).det()) / fact) for s in S]
    S = np.array([list(s) for s in S])
    return d, mlow, I.index(p3), Mlo, Mhi, S, np.array([v[0] for v in vols]), np.array([v[1] for v in vols]), len(V)


def mass_cell_fast(tri, gates, k, t0=1.5, tol=1.1, budget=4000, zbins=24, zratio=fmpq(23, 20), norm=R.NORM,
                   event=False, return_info=False, geom=None):
    """rigour.mass_cell(..., tangent=True) with bounds_batch: rigorous [lb, ub] on gamma(cell ∩ N_I ∩ {g23 >= t0}).
    Refinement as rigour (bisect the longest edge, in m-coordinates, of the simplex with the largest hi - lo), but
    in rounds: each round splits the top max(8, count // 16) simplices at once."""
    gm = geom if geom is not None else cell_geometry(tri, gates, k, norm=norm, event=event)
    if gm is None:
        return (0.0, 0.0, {}) if return_info else (0.0, 0.0)
    d, mlow, jp3, Mv_lo, Mv_hi, S, vlo, vhi, nV = gm
    P = Params(d, mlow, jp3, t0, zbins, zratio)
    Mlo = Mv_lo[S]; Mhi = Mv_hi[S]
    lo, hi = bounds_batch(Mlo, Mhi, vlo, vhi, P)
    S0 = len(lo); rounds = 0

    def sums():
        return float(up(math.fsum(hi))), float(dn(math.fsum(lo)))
    th, tl = sums()
    while th > tol * tl and len(lo) < budget:
        m = min(budget - len(lo), max(8, len(lo) // 16), len(lo))   # T400 fix: m <= number of simplices
        top = np.argpartition(-(hi - lo), m - 1)[:m] if m < len(lo) else np.arange(len(lo))
        Pl, Ph = Mlo[top], Mhi[top]
        Pm = 0.5 * Pl + 0.5 * Ph
        V = Pm.shape[1]
        ij = np.array(list(itertools.combinations(range(V), 2)))
        el = ((Pm[:, ij[:, 0]] - Pm[:, ij[:, 1]]) ** 2).sum(-1)
        e = ij[el.argmax(1)]
        r = np.arange(m)
        mlo = dn(dn(Pl[r, e[:, 0]] + Pl[r, e[:, 1]]) * 0.5)
        mhi = up(up(Ph[r, e[:, 0]] + Ph[r, e[:, 1]]) * 0.5)
        C1l, C1h, C2l, C2h = Pl.copy(), Ph.copy(), Pl.copy(), Ph.copy()
        C1l[r, e[:, 0]] = mlo; C1h[r, e[:, 0]] = mhi; C2l[r, e[:, 1]] = mlo; C2h[r, e[:, 1]] = mhi
        cvl = dn(vlo[top] * 0.5); cvh = up(vhi[top] * 0.5)
        Cl = np.concatenate([C1l, C2l]); Ch = np.concatenate([C1h, C2h])
        cvl2 = np.concatenate([cvl, cvl]); cvh2 = np.concatenate([cvh, cvh])
        clo, chi = bounds_batch(Cl, Ch, cvl2, cvh2, P)
        keep = np.ones(len(lo), bool); keep[top] = False
        Mlo = np.concatenate([Mlo[keep], Cl]); Mhi = np.concatenate([Mhi[keep], Ch])
        vlo = np.concatenate([vlo[keep], cvl2]); vhi = np.concatenate([vhi[keep], cvh2])
        lo = np.concatenate([lo[keep], clo]); hi = np.concatenate([hi[keep], chi])
        th, tl = sums(); rounds += 1
    ub = th; lb = max(0.0, tl)
    info = dict(d=d, mlow=mlow, vertices=nV, simplices0=S0, simplices=len(lo), rounds=rounds)
    return (lb, ub, info) if return_info else (lb, ub)


# ---------------------------------------------------------------- numba port of the per-(simplex, bin) set-up
@njit(cache=True, inline="always")
def _ndtr_err(x):
    return _up(LNDTR_ABS * SAFE * (1.0 + x * x))


@njit(cache=True)
def _Pn(xlo, xhi, upper, half, hm, coef, gh, ex, rel):
    """Scalar _P; ex = erfcx at dn(sqrt(xlo)) (upper) / up(sqrt(xhi)) (lower) when h is half-integer."""
    x = xhi if upper else xlo
    if not half:
        v = coef[hm - 1]
        for j in range(hm - 2, -1, -1):
            v = _up(_up(v * x) + coef[j]) if upper else _dn(_dn(v * x) + coef[j])
        return v
    if upper:
        sx = _up(np.sqrt(x)); v = _up(_up(gh[1] * ex) * (1 + 2 * rel))
    else:
        sx = _dn(np.sqrt(x)); v = _dn(_dn(gh[0] * ex) * (1 - 2 * rel))
    pw = sx
    for k in range(1, hm + 1):
        if upper:
            v = _up(v + _up(coef[k - 1] * pw)); pw = _up(pw * x)
        else:
            v = _dn(v + _dn(coef[k - 1] * pw)); pw = _dn(pw * x)
    return v


@njit(cache=True)
def _Cn(s, upper, half, hm):
    q = _up(2.0 / s) if upper else _dn(2.0 / s)
    v = 1.0
    for _ in range(hm):
        v = _up(v * q) if upper else _dn(v * q)
    if half:
        v = _up(v * _up(np.sqrt(q))) if upper else _dn(v * _dn(np.sqrt(q)))
    return 0.5 * v


@njit(cache=True, inline="always")
def _xiv(A, s):
    """Enclosure of A^2 s / 2."""
    p = A * A
    return _dn(_dn(_dn(p) * s) * 0.5), _up(_up(_up(p) * s) * 0.5)


@njit(cache=True)
def _prep1(sclo, sl, smin, smax, ypc, ylo, yhi, edges, mlow, half):
    """Arguments of log_ndtr (x0, xl, xh per bin; xs, xa on the last bin) and of erfcx (G factors)."""
    N = sclo.shape[0]
    nb = edges.shape[0] - 1
    xn = np.zeros((3, N, nb)); xe = np.zeros((4, N, nb)); xn2 = np.zeros((2, N)); xe2 = np.ones((2, N))
    for i in range(N):
        for b in range(nb):
            A = edges[b]; B = edges[b + 1]
            if mlow > 0:
                xn[0, i, b] = (0.5 * (A + B)) * ypc[i]
                xn[1, i, b] = min(_dn(A * ylo[i]), _dn(B * ylo[i]))
                xn[2, i, b] = max(_up(A * yhi[i]), _up(B * yhi[i]))
            if half:
                xa = _xiv(A, sclo[i]); xb = _xiv(B, sclo[i])
                xe[0, i, b] = _dn(np.sqrt(xa[0])); xe[1, i, b] = _up(np.sqrt(xb[1]))
                xa = _xiv(A, sl[i]); xb = _xiv(B, sl[i])
                xe[2, i, b] = _up(np.sqrt(xa[1])); xe[3, i, b] = _dn(np.sqrt(xb[0]))
        Al = edges[nb]
        if mlow > 0:
            xn2[0, i] = _up(Al * yhi[i]); xn2[1, i] = _dn(Al * ylo[i])
        if half:
            xa = _xiv(Al, smin[i]); xe2[0, i] = _dn(np.sqrt(xa[0]))
            xa = _xiv(Al, smax[i]); xe2[1, i] = _up(np.sqrt(xa[1]))
    return xn, xe, xn2, xe2


@njit(cache=True)
def _prep2(sclo, sl, smin, smax, T0, ypc, ylo, yhi, edges, mlow, half, hm, coef, gh, ls2p, Ln, Ex, Ln2, Ex2, rel):
    """Scalar port of _bounds_batch_np's set-up: Lu, Ll (2, N, nb) exponent bases, Gu, Gl (N, nb), kap, kal
    (2, N, nb), and the last-bin terms hinf, linf (N,)."""
    N = sclo.shape[0]
    nb = edges.shape[0] - 1
    Lu = np.empty((2, N, nb)); Ll = np.empty((2, N, nb)); Gu = np.empty((N, nb)); Gl = np.empty((N, nb))
    kap = np.zeros((2, N, nb)); kal = np.zeros((2, N, nb)); hinf = np.empty(N); linf = np.empty(N)
    llo = np.empty(nb); lhi = np.empty(nb)
    for b in range(nb):
        llo[b] = 0.5 * _dn(edges[b] * edges[b]); lhi[b] = 0.5 * _up(edges[b + 1] * edges[b + 1])
    for i in range(N):
        t0 = T0[i]
        neg = max(0.0, -ylo[i])
        for b in range(nb):
            A = edges[b]; B = edges[b + 1]
            if t0 < 0:
                E1u = _up(lhi[b] * (-t0)); E1l = _dn(llo[b] * (-t0))
            else:
                E1u = _up(-(llo[b] * t0)); E1l = _dn(-(lhi[b] * t0))
            # G factors: upper at s = sc_lo, lower at s = s_l
            s = sclo[i]
            xa = _xiv(A, s); xb = _xiv(B, s)
            BA = (_dn(_dn(_dn(_dn(B * B) - _up(A * A)) * s) * 0.5), _up(_up(_up(_up(B * B) - _dn(A * A)) * s) * 0.5))
            C = _Cn(s, True, half, hm)
            Gu[i, b] = _up(C * _up(_Pn(xa[0], xa[1], True, half, hm, coef, gh, Ex[0, i, b], rel)
                                   - _dn(_exp_dn(-BA[1], rel) * _Pn(xb[0], xb[1], False, half, hm, coef, gh, Ex[1, i, b], rel))))
            Gue = -xa[0]
            s = sl[i]
            xa = _xiv(A, s); xb = _xiv(B, s)
            BA = (_dn(_dn(_dn(_dn(B * B) - _up(A * A)) * s) * 0.5), _up(_up(_up(_up(B * B) - _dn(A * A)) * s) * 0.5))
            C = _Cn(s, False, half, hm)
            Gl[i, b] = _dn(C * max(0.0, _dn(_Pn(xa[0], xa[1], False, half, hm, coef, gh, Ex[2, i, b], rel)
                                            - _up(_exp_up(-BA[0], rel) * _Pn(xb[0], xb[1], True, half, hm, coef, gh, Ex[3, i, b], rel)))))
            Gle = -xa[1]
            if mlow > 0:
                BmA = _up(B - A)
                x0 = (0.5 * (A + B)) * ypc[i]
                xl = min(_dn(A * ylo[i]), _dn(B * ylo[i])); xh = max(_up(A * yhi[i]), _up(B * yhi[i]))
                e0 = _ndtr_err(x0)
                lp0l = _dn(Ln[0, i, b] - e0); lp0h = _up(Ln[0, i, b] + e0)
                exl = _dn(_dn(_dn(-0.5 * _up(x0 * x0)) - ls2p[1]) - lp0h)
                exh = _up(_up(_up(-0.5 * _dn(x0 * x0)) - ls2p[0]) - lp0l)
                rl = _exp_dn(exl, rel); rh = _exp_up(exh, rel)
                rt = np.exp(0.5 * exl + 0.5 * exh)
                if rt < 1e-200:
                    rt = 0.0
                Wx = max(_up(xh - x0), _up(x0 - xl))
                dlt = _up(max(_up(rh - rt), _up(rt - rl)) * Wx)
                cu = _up(lp0h + dlt); cu = _up(cu + _up(-(rt * x0))); cu = _up(cu + _up(_up(rt * BmA) * neg)); cu = _up(mlow * cu)
                kap[0, i, b] = max(0.0, _dn(_dn(mlow * rt) * B)); kap[1, i, b] = _up(_up(mlow * rt) * B)
                L1 = _dn(Ln[1, i, b] - _ndtr_err(xl))
                eh = _ndtr_err(xh)
                L2 = _dn(Ln[2, i, b] - eh); L2h = _up(Ln[2, i, b] + eh)
                den0 = _dn(xh - xl); den1 = _up(xh - xl)
                bt = 0.0; eb = 0.0
                if den0 > 0:
                    num0 = _dn(L2 - L1); num1 = _up(L2 - L1)
                    bl = _dn(num0 / den1) if num0 >= 0 else _dn(num0 / den0)
                    bh = _up(num1 / den0) if num1 >= 0 else _up(num1 / den1)
                    bt = bl if bl > 1e-200 else 0.0
                    eb = max(0.0, max(_up(bh - bt), _up(bt - bl)))
                cl = _dn(L1 + _dn(-(bt * xl))); cl = _dn(cl - _up(eb * den1)); cl = _dn(cl - _up(_up(bt * BmA) * neg)); cl = _dn(mlow * cl)
                kal[0, i, b] = max(0.0, _dn(_dn(mlow * bt) * A)); kal[1, i, b] = _up(_up(mlow * bt) * A)
                lsup = _up(mlow * L2h); lin = _dn(mlow * L1)
                Lu[0, i, b] = _up(_up(E1u + Gue) + lsup); Lu[1, i, b] = _up(_up(E1u + cu) + Gue)
                Ll[0, i, b] = _dn(_dn(E1l + Gle) + lin); Ll[1, i, b] = _dn(_dn(E1l + cl) + Gle)
            else:
                Lu[0, i, b] = _up(E1u + Gue); Ll[0, i, b] = _dn(E1l + Gle)
                Lu[1, i, b] = np.inf; Ll[1, i, b] = -np.inf
        # last bin [A, inf): upper at s_min, lower at s_max
        Al = edges[nb]
        xa = _xiv(Al, smin[i])
        g1 = _up(_Cn(smin[i], True, half, hm) * _Pn(xa[0], xa[1], True, half, hm, coef, gh, Ex2[0, i], rel)); e1 = -xa[0]
        xa = _xiv(Al, smax[i])
        g2 = _dn(_Cn(smax[i], False, half, hm) * _Pn(xa[0], xa[1], False, half, hm, coef, gh, Ex2[1, i], rel)); e2 = -xa[1]
        if mlow > 0:
            xs = _up(Al * yhi[i])
            lsupi = 0.0 if yhi[i] >= 0 else _up(mlow * _up(Ln2[0, i] + _ndtr_err(xs)))
            hinf[i] = _up(_exp_up(_up(e1 + lsupi), rel) * g1)
            xa_ = _dn(Al * ylo[i])
            li = _dn(mlow * _dn(Ln2[1, i] - _ndtr_err(xa_)))
            linf[i] = 0.0 if ylo[i] < 0 else _dn(_exp_dn(_dn(e2 + li), rel) * g2)
        else:
            hinf[i] = _up(_exp_up(e1, rel) * g1); linf[i] = _dn(_exp_dn(e2, rel) * g2)
    return Lu, Ll, Gu, Gl, kap, kal, hinf, linf


def _bounds_batch(Mlo, Mhi, vlo, vhi, P, mc=None):
    Mlo = np.asarray(Mlo, float); Mhi = np.asarray(Mhi, float)
    c = simplex_consts(Mlo, Mhi, P.jp3, mc)
    T0 = c["tau"][0].min(1)                                             # split point (exact float) <= tau_i
    tshlo = np.ascontiguousarray(np.maximum(0.0, dn(c["tau"][0] - T0[:, None])))
    tshhi = np.ascontiguousarray(up(c["tau"][1] - T0[:, None]))
    yplo = np.ascontiguousarray(c["yplo"]); yphi = np.ascontiguousarray(c["yphi"])
    ylo = yplo.min(1); yhi = yphi.max(1)
    sclo = c["sc"][0]; sl = up(c["sc"][1] + c["R2"]); smin = c["smin"]; smax = c["smax"]
    mlow = P.mlow
    xn, xe, xn2, xe2 = _prep1(sclo, sl, smin, smax, c["ypc"], ylo, yhi, P.edges, mlow, P.half)
    Ln = log_ndtr(xn) if mlow else xn; Ln2 = log_ndtr(xn2) if mlow else xn2
    Ex = erfcx(xe) if P.half else xe; Ex2 = erfcx(xe2) if P.half else xe2
    Lu, Ll, Gu, Gl, kap, kal, hinf, linf = _prep2(sclo, sl, smin, smax, T0, c["ypc"], ylo, yhi, P.edges, mlow, P.half,
                                                  P.hm, P.coef, P.ghv, np.array(P.ls2p), Ln, Ex, Ln2, Ex2, REL_T)
    llo = np.ascontiguousarray(0.5 * dn(P.edges[:-1] ** 2)); lhi = np.ascontiguousarray(0.5 * up(P.edges[1:] ** 2))
    hsum, lsum, nex = _bins(tshlo, tshhi, yplo, yphi, llo, lhi, mlow, kap, kal, Lu, Ll, Gu, Gl, REL_T, WMAX, EPS)
    hs = up(hsum + hinf)
    ls = np.maximum(0.0, dn(lsum + linf))
    hi = up(up(P.c0[1] * np.asarray(vhi)) * hs)
    lo = dn(dn(P.c0[0] * np.asarray(vlo)) * ls)
    STATS["exact_bins"] += int(nex.sum()); STATS["simplices"] += len(hs)
    return lo, hi

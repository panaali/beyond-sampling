"""Tube / thin-wedge estimator of the error of M_16,10 (see README.md).

Exact co-area identity used everywhere: fibre input space by straight segments in a tie direction.
  * 2-3 face: x_a = x_b = u (a<b unordered), one position c holds the max x_c > u, the other 7 are < u.
    Segment: x_a = u + s/2, x_b = u - s/2, |s| < S = min(2(x_c-u)/3, 2(u-v4)).
    Then (v2-v3) < (v1-v2) on the segment, and every input with v2-v3 < v1-v2 lies on exactly one segment.
  * 1-2 face: x_a = x_b = w, the other 8 < w. Segment x_a = w + t/2, x_b = w - t/2, |t| < 2(w - v3)/3.
    Covers every input with v1-v2 <= v2-v3.
  * Blind region B = {v2 <= 0} (<=1 positive input), mass exactly 11/1024, cut out of both families.
Jacobian of (x_a,x_b)->(u,s) is 1 and phi(u+s/2)phi(u-s/2) = phi(u)^2 exp(-s^2/4), so
  P(err, not B) = Z23 * E_face23[F] + Z12 * E_face12[F],   F = int_{segment, not B} exp(-s^2/4) 1[err(s)] ds,
with Z (model-free, 1-D quadrature) the total face density and E over the normalised face measure.
The network is exactly piecewise linear along a segment; F is computed by walking its linear cells
(one forward + one tangent (JVP) pass per cell), solving exactly where the argmax changes, and integrating
exp(-s^2/4) with erf. Walking stops after K cells per side (the wedge approximation is K=1 'first cell').
Face coordinates are integrated with a deterministic (unscrambled Sobol) rule, positions enumerated.
"""
import numpy as np
from scipy.special import ndtr, ndtri, erf
from scipy.stats import qmc

T, H = 10, 16
SQPI = np.sqrt(np.pi)


# ------------------------------------------------------------------ model-free face measures (1-D quadrature)
_g = np.linspace(-9, 9, 200001)
_phi = np.exp(-_g ** 2 / 2) / np.sqrt(2 * np.pi)
_Phi = ndtr(_g)
_Q = ndtr(-_g)


def _table(dens):
    c = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(_g))])
    return c[-1], c / c[-1]


Z23_1, CDF23 = _table(_phi ** 2 * _Q * _Phi ** 7)     # per (pair, c) combo
Z12_1, CDF12 = _table(_phi ** 2 * _Phi ** 8)          # per pair
Z23 = 45 * 8 * Z23_1
Z12 = 45 * Z12_1
P_BLIND = 11 / 1024
PAIRS = [(a, b) for a in range(T) for b in range(a + 1, T)]
COMBO23 = [(a, b, c) for (a, b) in PAIRS for c in range(T) if c not in (a, b)]   # 360


def _inv(cdf, v):
    return np.interp(v, cdf, _g)


def _trunc_below(u, v):    # N(0,1) conditioned < u
    return ndtri(np.clip(v * ndtr(u), 1e-300, 1))


def _trunc_above(u, v):    # N(0,1) conditioned > u
    return -ndtri(np.clip(v * ndtr(-u), 1e-300, 1))


SHIFT = (np.arange(1, 12) * 0.6180339887498949) % 1.0   # fixed deterministic shift (Kronecker)


def _qmc(n, d, skip=0):
    s = qmc.Sobol(d, scramble=False)
    if skip:
        s.fast_forward(skip)
    return s.random(n)


def faces23(n, shift=SHIFT):
    """n face points (deterministic). Positions cycle through the 360 combos (exactly stratified when 360|n)."""
    U = _qmc(n, 10)
    if shift is not None:
        U = (U + shift[:10]) % 1.0
    combo = np.array(COMBO23)[(U[:, 9] * 360).astype(int)]
    u = _inv(CDF23, U[:, 0])
    xc = _trunc_above(u, U[:, 1])
    rest = _trunc_below(u[:, None], U[:, 2:9])
    x0 = np.empty((n, T)); d = np.zeros((n, T))
    for k in range(n):
        a, b, c = combo[k]
        others = [i for i in range(T) if i not in (a, b, c)]
        x0[k, a] = x0[k, b] = u[k]; x0[k, c] = xc[k]; x0[k, others] = rest[k]
        d[k, a], d[k, b] = 0.5, -0.5
    v4 = rest.max(1)
    S = np.minimum(2 * (xc - u) / 3, 2 * (u - v4))
    lo = np.maximum(0, -2 * u)              # blind cut: v2 = u + |s|/2 > 0
    return dict(x0=x0, d=d, lo=lo, hi=S, tpos=combo[:, 0], tneg=combo[:, 1], Z=Z23)


def faces12(n, shift=SHIFT):
    U = _qmc(n, 10)
    if shift is not None:
        U = (U + shift[:10]) % 1.0
    pr = np.array(PAIRS)[(U[:, 9] * 45).astype(int)]
    w = _inv(CDF12, U[:, 0])
    rest = _trunc_below(w[:, None], U[:, 1:9])
    x0 = np.empty((n, T)); d = np.zeros((n, T))
    for k in range(n):
        a, b = pr[k]
        others = [i for i in range(T) if i not in (a, b)]
        x0[k, a] = x0[k, b] = w[k]; x0[k, others] = rest[k]
        d[k, a], d[k, b] = 0.5, -0.5
    v3 = rest.max(1)
    S = 2 * (w - v3) / 3
    hi = np.minimum(S, 2 * w)               # v2 = w - |t|/2 > 0
    # t>0: a is the max -> target b ; t<0: target a
    return dict(x0=x0, d=d, lo=np.zeros(n), hi=hi, tpos=pr[:, 1], tneg=pr[:, 0], Z=Z12)


def blind_points(n, shift=SHIFT):
    """Deterministic points of the blind region with their exact masses.
    0 positives (mass 1/1024) and exactly one positive at c (mass 1/1024 each)."""
    U = _qmc(n, T + 1)
    if shift is not None:
        U = (U + shift[:T + 1]) % 1.0
    z = -ndtri(np.clip(U[:, :T] / 2, 1e-300, 1))      # half-normal magnitude, |z|
    x = -z
    c = (U[:, T] * 11).astype(int)                        # 0..9 = position of the positive, 10 = none
    m = c < 10
    x[np.where(m)[0], c[m]] *= -1
    return x


# ------------------------------------------------------------------ exact piecewise-linear walk
def fwd_jvp(model, x, dx):
    Wi, Wh, Wo = model
    n = len(x)
    h = np.zeros((n, Wh.shape[0])); dh = np.zeros_like(h)
    P = np.empty((n, T, Wh.shape[0])); dP = np.empty_like(P)
    for t in range(T):
        p = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        dp = dh @ Wh.T + dx[:, t:t + 1] * Wi[None]
        P[:, t], dP[:, t] = p, dp
        g = p > 0
        h, dh = p * g, dp * g
    return h @ Wo.T, dh @ Wo.T, P, dP


def W(a, b):
    return SQPI * (erf(b / 2) - erf(a / 2))


def walk_side(model, x0, d, lo, hi, tgt, K, Wf=None, extrap=False):
    Wf = W if Wf is None else Wf
    """Integrate exp(-r^2/4) 1[argmax != tgt] over r in [lo, hi] along x0 + r d, walking <= K linear cells.
    Returns (error mass found, mass not walked, cells used per point)."""
    n = len(x0)
    err = np.zeros(n); r = lo.copy(); active = r < hi; cells = np.zeros(n)
    idx = np.arange(n)
    # FLOP accounting in forward-pass units: a cell recomputes value+tangent only from the time step whose
    # gate flipped (earlier steps are unchanged); the first cell starts at the first step where d != 0.
    tstart = (np.abs(d) > 0).argmax(1).astype(float)
    for _ in range(K):
        ii = idx[active]
        if len(ii) == 0:
            break
        rc = r[ii]; scale = 1e-9 * (1 + np.abs(x0[ii]).max(1))
        re = rc + scale
        L, dL, P, dP = fwd_jvp(model, x0[ii] + re[:, None] * d[ii], d[ii])
        cells[ii] += 2 * (T - tstart[ii]) / T
        with np.errstate(divide="ignore", invalid="ignore"):
            root = -P / dP
        root[~(root > 0)] = np.inf
        rf = root.reshape(len(ii), -1)
        tstart[ii] = rf.argmin(1) // P.shape[2]
        rn = np.minimum(re + rf.min(1), hi[ii])
        if extrap and _ == K - 1:
            rn = hi[ii].copy()          # last cell: extrapolate the linear map to the segment end
        t = tgt[ii]
        Lt = L[np.arange(len(ii)), t][:, None]; dLt = dL[np.arange(len(ii)), t][:, None]
        D0 = Lt - L; dD = dLt - dL               # D_k(r) = D0 + (r - re) dD >= 0 needed for all k != t
        D0[np.arange(len(ii)), t] = 1.0; dD[np.arange(len(ii)), t] = 0.0
        A = rc.copy(); B = rn.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = re[:, None] - D0 / dD
        pos = dD > 0; neg = dD < 0; zer = dD == 0
        A = np.maximum(A, np.where(pos, rr, -np.inf).max(1))
        B = np.minimum(B, np.where(neg, rr, np.inf).min(1))
        never = (zer & (D0 < 0)).any(1)
        good = np.where((B > A) & ~never, Wf(A, B), 0.0)
        err[ii] += Wf(rc, rn) - good
        r[ii] = rn
        active[ii] = rn < hi[ii] - 1e-12
    rem = np.where(active, Wf(r, hi), 0.0)
    return err, rem, cells


def _cos8(t):
    return 35 * t / 128 + 7 * np.sin(2 * t) / 32 + 7 * np.sin(4 * t) / 128 + np.sin(6 * t) / 96 + np.sin(8 * t) / 1024


C_RAD = 2 ** 0.5 * 24 / 11.631728396567448     # sqrt2 Gamma(5)/Gamma(9/2)


def Wrad(a, b):
    """Radially integrated weight: C * int_a^b (1+k^2/2)^-5 dk (cone measure along slope k)."""
    G = lambda k: np.sqrt(2) * _cos8(np.arctan(k / np.sqrt(2)))
    return C_RAD * (G(b) - G(a))


def F_faces(model, fc, K, radial=False, extrap=False):
    """F for each face point: error mass on both sides of the tie, <= K cells per side.
    radial=True: integrate the radius of the face point analytically (homogeneity -> error set is a cone)."""
    x0, lo, hi, Wf = fc["x0"], fc["lo"], fc["hi"], W
    if radial:
        rho = np.linalg.norm(x0, axis=1)
        x0, lo, hi, Wf = x0 / rho[:, None], lo / rho, hi / rho, Wrad
    e1, r1, c1 = walk_side(model, x0, fc["d"], lo, hi, fc["tpos"], K, Wf, extrap)
    e2, r2, c2 = walk_side(model, x0, -fc["d"], lo, hi, fc["tneg"], K, Wf, extrap)
    prefix = (np.abs(fc["d"]) > 0).argmax(1) / T          # shared forward prefix (value only)
    return e1 + e2, r1 + r2, c1 + c2 + prefix


# ------------------------------------------------------------------ hybrid: exact walk near the tie, one-point quadrature beyond
_TH = np.linspace(0, np.pi / 2, 20001)
_GTH = np.sqrt(2) * _cos8(_TH)


def Grad(k):
    return C_RAD * np.sqrt(2) * _cos8(np.arctan(k / np.sqrt(2)))


def Ginv(g):
    th = np.interp(g / C_RAD, _GTH, _TH)
    return np.sqrt(2) * np.tan(th)


def forward_from(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def F_hybrid(model, fc, k0, v_tail):
    """Radial-integrated F: exact walk on slopes [lo, min(hi, k0)] on each side; on [k0, hi] one evaluation at the
    slope with weight-quantile v_tail (deterministic quadrature node), weighted by the tail's mass.
    v_tail: (n, 2) numbers in (0,1)."""
    rho = np.linalg.norm(fc["x0"], axis=1)
    x0, lo, hi = fc["x0"] / rho[:, None], fc["lo"] / rho, fc["hi"] / rho
    near_hi = np.minimum(hi, k0)
    tot = np.zeros(len(x0)); cost = (np.abs(fc["d"]) > 0).argmax(1) / T
    tmin = cost.copy()
    for side, sg, tg in ((0, 1.0, fc["tpos"]), (1, -1.0, fc["tneg"])):
        e, _, c = walk_side(model, x0, sg * fc["d"], lo, near_hi, tg, 200, Wrad)
        tot += e; cost += c
        a = np.maximum(lo, k0); m = hi > a
        if m.any():
            ga, gb = Grad(a[m]), Grad(hi[m])
            k = Ginv(ga + v_tail[m, side] * (gb - ga))
            x = x0[m] + (sg * k)[:, None] * fc["d"][m]
            wrong = forward_from(model, x).argmax(1) != tg[m]
            tot[m] += (gb - ga) * wrong
            cost[m] += (T - T * tmin[m]) / T
    return tot, cost

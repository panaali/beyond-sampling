"""T418: M_16,10 unrolled as a torch module on homogeneity-reduced u-coordinates, for auto_LiRPA / alpha-beta-CROWN.
Independent of T414's code: own coordinates, own forward, own slab sampler and charges; only the weights are shared
(m16/common.py, read-only). Coordinates (T414 README): x = g y, g = x_p2 - x_p3 >= t0, y_p3 = u_r, y_p2 = u_r + 1,
y_p1 = u_r + 1 + u_a, y_l = u_r - u_l (lows in index order), u_a, u_l >= 0. Output of UNet: the 9 margins
L_j - L_p3 (j != p3, increasing j); the event is all margins <= 0, a box is pruned iff some lower bound is > 0."""
import importlib.util
import math
import os

import numpy as np
import torch as th
from torch import nn

# load m16/common.py by path (putting m16/ on sys.path would shadow alpha-beta-CROWN's `state` package with m16/state.py)
_spec = importlib.util.spec_from_file_location("m16_common", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common.py"))
_common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_common)
_Wi, _Wh, _Wo = _common.Wi, _common.Wh, _common.Wo

D, H, T0, U = 10, 16, 2.5, 4.0
dt = th.float64
WI, WH, WO = (th.as_tensor(np.asarray(w, float), dtype=dt) for w in (_Wi, _Wh, _Wo))


def ymap(tri):
    """y = A u + c, A (10, 9), c (10,)"""
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]
    A = th.zeros(D, 9, dtype=dt); c = th.zeros(D, dtype=dt)
    A[[p1, p2, p3], 0] = 1; c[p1] = 1; c[p2] = 1; A[p1, 1] = 1
    for k, l in enumerate(lows):
        A[l, 0] = 1; A[l, 2 + k] = -1
    return A, c


def root(tri):
    lo = th.zeros(9, dtype=dt); hi = th.full((9,), U, dtype=dt); lo[0] = -U
    return lo, hi


class UNet(nn.Module):
    """u (N, 9) -> 9 margins. Step t: h = relu(Wh h + In_t(u)), In_t(u) = Wi * y_t(u) as a Linear(9, 16) with bias."""
    def __init__(self, tri):
        super().__init__()
        A, c = ymap(tri); p3 = tri[2]; others = [j for j in range(D) if j != p3]
        self.inp = nn.ModuleList()
        for t in range(D):
            lin = nn.Linear(9, H).double(); lin.weight.data = WI[:, None] * A[t][None]; lin.bias.data = WI * c[t]
            self.inp.append(lin)
        self.rec = nn.Linear(H, H, bias=False).double(); self.rec.weight.data = WH.clone()
        self.out = nn.Linear(H, 9, bias=False).double(); self.out.weight.data = WO[others] - WO[p3]

    def forward(self, u):
        h = th.relu(self.inp[0](u))
        for t in range(1, D):
            h = th.relu(self.rec(h) + self.inp[t](u))
        return self.out(h)


def margins_x(x, p3):
    """reference forward on raw inputs x (N, 10), straight from the RNN recurrence: 9 margins L_j - L_p3"""
    h = th.zeros(x.shape[0], H, dtype=dt)
    for t in range(D):
        h = th.relu(h @ WH.T + x[:, t:t + 1] * WI)
    L = h @ WO.T; others = [j for j in range(D) if j != p3]
    return L[:, others] - L[:, p3:p3 + 1]


def slab_sample(tri, n, rng):
    """exact samples of x conditioned on the triple's slab {x_p1 >= x_p2 >= x_p3 + t0, lows <= x_p3}, by sequential
    inverse CDF: x_p3 ~ phi Phi^7 sf(.+t0)^2-weight (marginal of the slab), x_p2 | x_p3, x_p1 | x_p2, lows | x_p3.
    The marginal of r = x_p3 is phi(r) Phi(r)^7 J(r), J(r) = int_{r+t0} phi(s) sf(s) ds = sf(r+t0)^2 / 2."""
    from scipy import stats
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]
    g = np.linspace(-7, 5, 400001); lw = stats.norm.logpdf(g) + 7 * stats.norm.logcdf(g) + 2 * stats.norm.logsf(g + T0)
    cdf = np.cumsum(np.exp(lw - lw.max())); cdf /= cdf[-1]; r = np.interp(rng.uniform(size=n), cdf, g)
    x = np.zeros((n, D)); x[:, p3] = r
    # x_p2 | r has density prop. to phi(s) sf(s) on [r + t0, inf): inverse CDF of sf(s)^2 / 2 (closed form)
    a = stats.norm.sf(r + T0); x[:, p2] = stats.norm.isf(a * np.sqrt(rng.uniform(size=n)))
    x[:, p1] = stats.norm.isf(rng.uniform(size=n) * stats.norm.sf(x[:, p2]))
    x[:, lows] = stats.norm.ppf(rng.uniform(size=(n, 7)) * stats.norm.cdf(r)[:, None])
    return x


def x_to_u(x, tri):
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]
    g = x[:, p2] - x[:, p3]; y = x / g[:, None]
    return np.stack([y[:, p3], y[:, p1] - y[:, p2]] + [y[:, p3] - y[:, l] for l in lows], 1), g


# ---------------- charges (own implementations) ----------------
def minsq(lo, hi):
    """rigorous lower bound of min |y(u)|^2 over u-boxes (N, 9): |y|^2 = r^2 + (r+1)^2 + (r+1+a)^2 + sum (r - l_k)^2.
    For fixed r the other coordinates minimise independently (clamp), the result G(r) is convex in r; minimise G by
    bisection on the derivative, then take the tangent lower bound at the bracket (rigorous)."""
    def G(r):
        a = th.minimum(th.maximum(-(r + 1), lo[:, 1]), hi[:, 1]); l = th.maximum(th.minimum(r[:, None], hi[:, 2:]), lo[:, 2:])
        ya = r + 1 + a; yl = r[:, None] - l
        return r ** 2 + (r + 1) ** 2 + ya ** 2 + (yl ** 2).sum(1), 2 * r + 2 * (r + 1) + 2 * ya + 2 * yl.sum(1)
    a_, b_ = lo[:, 0].clone(), hi[:, 0].clone()
    for _ in range(64):
        m = (a_ + b_) / 2; _, d = G(m); neg = d < 0; a_ = th.where(neg, m, a_); b_ = th.where(neg, b_, m)
    Ga, da = G(a_); Gb, db = G(b_)
    # convexity: G(r) >= G(a_) + G'(a_)(r - a_) and >= G(b_) + G'(b_)(r - b_); on [a_, b_] min >= max of the two
    # tangent lines' minima over the bracket, which is >= min(Ga + da*(b_-a_), Gb - db*(b_-a_)) (da <= 0 <= db)
    bnd = th.minimum(Ga + da.clamp(max=0) * (b_ - a_), Gb - db.clamp(min=0) * (b_ - a_))
    Glo, dlo = G(lo[:, 0]); Ghi, dhi = G(hi[:, 0])
    return th.where(dlo >= 0, Glo, th.where(dhi <= 0, Ghi, bnd)).clamp_min(0.5)


def radial_charge(lo, hi):
    """T414's pre-registered rigorous charge, re-implemented: vol * 16 (2 pi)^-5 m^-5 Gamma(5, m t0^2 / 2)"""
    m = minsq(lo, hi); w = m * T0 ** 2 / 2
    gam = 24 * th.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
    return (hi - lo).prod(1) * 16 * (2 * math.pi) ** -5 * m ** -5 * gam


_GL = {}


def _gl(n):
    if n not in _GL:
        z, w = np.polynomial.legendre.leggauss(n); _GL[n] = (th.as_tensor(z, dtype=dt), th.as_tensor(w, dtype=dt))
    return _GL[n]


def _Phi(z):
    return 0.5 * th.special.erfc(-z / math.sqrt(2))


def quad_mass(lo, hi, nr=4, nq=12, ng=24, gmax=14.0):
    """near-exact slab mass of u-boxes (float, not a certificate):
    mass = int_{g >= t0} int_{u_r} g phi(g u_r) phi(g (u_r + 1)) [Phi(g(u_r+1+a1)) - Phi(g(u_r+1+a0))]
           prod_k [Phi(g(u_r - l0_k)) - Phi(g(u_r - l1_k))] du_r dg
    (the u_a and u_l integrals are exact; Jacobian g^9 = g * g^8, each g^1 absorbed by one Phi difference).
    u_r: composite Gauss-Legendre, nr panels x nq nodes; g on [t0, gmax] with ng nodes on a sqrt-stretched grid."""
    N = lo.shape[0]; z, w = _gl(nq)
    edges = lo[:, :1] + (hi[:, :1] - lo[:, :1]) * th.linspace(0, 1, nr + 1, dtype=dt)[None]      # (N, nr+1)
    half = (edges[:, 1:] - edges[:, :-1]) / 2; mid = (edges[:, 1:] + edges[:, :-1]) / 2
    ur = (mid[:, :, None] + half[:, :, None] * z).reshape(N, -1); wr = (half[:, :, None] * w).reshape(N, -1)
    zg, wg = _gl(ng); s = (zg + 1) / 2                                                            # s in (0,1)
    gg = T0 + (gmax - T0) * s ** 2; wgg = wg / 2 * (gmax - T0) * 2 * s                            # g = t0 + (gmax-t0) s^2
    out = th.zeros(N, dtype=dt)
    for gi in range(ng):
        g = gg[gi]
        f = g * th.exp(-0.5 * g * g * (ur ** 2 + (ur + 1) ** 2)) / (2 * math.pi)
        f = f * (_Phi(g * (ur + 1 + hi[:, 1:2])) - _Phi(g * (ur + 1 + lo[:, 1:2])))
        for k in range(7):
            f = f * (_Phi(g * (ur - lo[:, 2 + k:3 + k])) - _Phi(g * (ur - hi[:, 2 + k:3 + k])))
        out = out + wgg[gi] * (f * wr).sum(1)
    return out

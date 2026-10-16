"""Certified probabilistic branch-and-bound for M_16,10 accuracy.

Regions: root (a, b) = "x_a is the max, x_b is the 2nd" (90 roots, target = b, constant on the root).
Inside a root we use anchored gap coordinates w = (v, r, e_1..e_8):
    x_b = v,  x_a = v + r,  x_j = v - e_j  (j = the 8 other positions, increasing order)
with r >= 0, e_j >= 0. Every ordering constraint of the root is then a box face, so a region is an
axis-aligned box in w and its Gaussian measure is the 1-D integral
    mu = int_{v} phi(v) [Phi(v+r_hi)-Phi(v+r_lo)] prod_j [Phi(v-e_lo_j)-Phi(v-e_hi_j)] dv
(Gauss-Legendre, deterministic, model-free). Bounds on logit_b - logit_j over the box: forward symbolic
(linear) propagation for intermediate pre-activation bounds + backward CROWN for the 9 margins.
"""
import sys
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, "/home/user/clean-neurons/m16")
from common import Wi as Wi0, Wh as Wh0, Wo as Wo0  # noqa: E402

T, D = 10, 10
ROOTS = [(a, b) for a in range(T) for b in range(T) if a != b]
DOM = dict(v=(-3.0, 4.5), r=(0.0, 6.0), e=(0.0, 8.0))   # truncated domain; outside mass -> undecided


def root_matrix(a, b):
    """x = M w."""
    M = np.zeros((T, D))
    M[:, 0] = 1.0
    M[a, 1] = 1.0
    rest = [j for j in range(T) if j not in (a, b)]
    for k, j in enumerate(rest):
        M[j, 2 + k] = -1.0
    return M


MS = np.array([root_matrix(a, b) for a, b in ROOTS])          # (90, T, D)
TGT = np.array([b for a, b in ROOTS])


def root_boxes():
    lo = np.array([DOM["v"][0], DOM["r"][0]] + [DOM["e"][0]] * 8)
    hi = np.array([DOM["v"][1], DOM["r"][1]] + [DOM["e"][1]] * 8)
    return np.repeat(lo[None], 90, 0), np.repeat(hi[None], 90, 0)


_GL = {n: leggauss(n) for n in (16, 24, 48, 96)}


def measure(wl, wu, n=24):
    """Gaussian measure of boxes in w-space (any root: it does not depend on (a,b)). Vectorised.
    FLOPs ~ n * (2*9 Phi + few) ~ n*9*2*~25."""
    x, wq = _GL[n]
    vl, vu = wl[:, :1], wu[:, :1]
    v = 0.5 * (vl + vu) + 0.5 * (vu - vl) * x[None]                  # (N,n)
    f = np.exp(-0.5 * v * v) / np.sqrt(2 * np.pi)
    f = f * (ndtr(v + wu[:, 1:2]) - ndtr(v + wl[:, 1:2]))
    for k in range(2, D):
        # Phi(v-elo)-Phi(v-ehi) = Phi(ehi-v)-Phi(elo-v): use the form with smaller cancellation
        f = f * (ndtr(wu[:, k:k + 1] - v) - ndtr(wl[:, k:k + 1] - v))
    return 0.5 * (vu - vl)[:, 0] * (f * wq[None]).sum(1)


def _relax(l, u):
    """CROWN relaxation slopes for h=relu(z), l<=z<=u: h >= al*z, h <= au*z + bu."""
    act = l >= 0
    unst = (l < 0) & (u > 0)
    au = np.where(act, 1.0, 0.0)
    bu = np.zeros_like(l)
    lam = np.where(unst, u / np.where(unst, u - l, 1.0), 0.0)
    au = np.where(unst, lam, au)
    bu = np.where(unst, -lam * l, bu)
    al = np.where(act, 1.0, np.where(unst & (u > -l), 1.0, 0.0))
    return al, au, bu


def _conc_lo(c, k, wc, wr):
    """min over box of c.w + k  (c: (N,K,D))."""
    return k + np.einsum("nkd,nd->nk", c, wc) - np.einsum("nkd,nd->nk", np.abs(c), wr)


def bounds(root, wl, wu, model=None, inter="fwd"):
    """Lower and upper bounds of margins m_j = logit_target - logit_j (j != target) over each box.
    Returns lb (N,T), ub (N,T) with +inf/-inf... at j=target set to +inf, plus final lower-bound coefs."""
    Wi, Wh, Wo = model if model is not None else (Wi0, Wh0, Wo0)
    N = len(root)
    M = MS[root]                                    # (N,T,D)
    wc, wr = 0.5 * (wl + wu), 0.5 * (wu - wl)
    Whp, Whn = np.maximum(Wh, 0), np.minimum(Wh, 0)
    # ---- intermediate bounds: forward symbolic propagation of linear lower/upper functions of w
    Ls, Us = [], []                                  # bounds of z_t, t=1..T
    hl_c = np.zeros((N, 16, D)); hl_k = np.zeros((N, 16))
    hu_c = np.zeros((N, 16, D)); hu_k = np.zeros((N, 16))
    for t in range(T):
        xin = Wi[None, :, None] * M[:, t, None, :]                    # (N,16,D)
        zl_c = Whp @ hl_c + Whn @ hu_c + xin
        zl_k = hl_k @ Whp.T + hu_k @ Whn.T
        zu_c = Whp @ hu_c + Whn @ hl_c + xin
        zu_k = hu_k @ Whp.T + hl_k @ Whn.T
        l = _conc_lo(zl_c, zl_k, wc, wr)
        u = -_conc_lo(-zu_c, -zu_k, wc, wr)
        if inter == "crown" and t > 0:
            l2, u2 = _crown_z(t + 1, M, wc, wr, Ls, Us, Wi, Wh)
            l, u = np.maximum(l, l2), np.minimum(u, u2)
        Ls.append(l); Us.append(u)
        al, au, bu = _relax(l, u)
        hl_c = al[..., None] * zl_c; hl_k = al * zl_k
        hu_c = au[..., None] * zu_c; hu_k = au * zu_k + bu
    # ---- final margins: backward CROWN using intermediate bounds
    tg = TGT[root]
    C = Wo[tg][:, None, :] - Wo[None, :, :]                           # (N,T,16) rows j
    A = np.concatenate([C, -C], 1)                                     # lower of m, lower of -m
    coef, const = _backward(A, T, M, Ls, Us, Wi, Wh)
    lo = _conc_lo(coef, const, wc, wr)
    lb, ub = lo[:, :T], -lo[:, T:]
    idx = np.arange(N)
    lb[idx, tg] = np.inf; ub[idx, tg] = np.inf
    return lb, ub, coef[:, :T]


def _backward(A, t, M, Ls, Us, Wi, Wh):
    """A: coefficients on h_t (N,K,16). Returns linear lower bound (coef on w, const)."""
    N, K, _ = A.shape
    coef = np.zeros((N, K, D)); const = np.zeros((N, K))
    for s in range(t, 0, -1):
        al, au, bu = _relax(Ls[s - 1], Us[s - 1])                      # relax h_s = relu(z_s)
        Ap, An = np.maximum(A, 0), np.minimum(A, 0)
        Az = Ap * al[:, None] + An * au[:, None]
        const += np.einsum("nkh,nh->nk", An, bu)
        coef += (Az @ Wi)[..., None] * M[:, s - 1, None, :]            # z_s = Wh h_{s-1} + Wi x_{s-1}
        A = Az @ Wh
    return coef, const


def _crown_z(t, M, wc, wr, Ls, Us, Wi, Wh):
    """CROWN bounds of z_t using bounds of z_1..z_{t-1}."""
    N = len(M)
    I = np.broadcast_to(np.eye(16), (N, 16, 16))
    A = np.concatenate([I, -I], 1) @ Wh                                # coef on h_{t-1}
    coef, const = _backward(A, t - 1, M, Ls, Us, Wi, Wh)
    coef = coef + np.concatenate([Wi, -Wi])[None, :, None] * M[:, t - 1, None, :]
    lo = _conc_lo(coef, const, wc, wr)
    return lo[:, :16], -lo[:, 16:]


def forward_x(x, model=None):
    Wi, Wh, Wo = model if model is not None else (Wi0, Wh0, Wo0)
    h = np.zeros((len(x), 16))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


# FLOP model per region (multiply-add = 2 FLOPs); RNN pass = 6000 FLOPs
FLOP_FWD = T * (4 * 16 * 16 * (D + 1) * 2 + 4 * 16 * D * 2)          # forward symbolic
FLOP_BWD = T * (2 * T * 16 * 16 * 2 + 2 * T * 16 * D * 2)             # backward, 20 rows
FLOP_CROWNZ = sum(s * (32 * 16 * 16 * 2 + 32 * 16 * D * 2) for s in range(1, T))
FLOP_MEAS = 24 * (9 * 2 * 30 + 20)

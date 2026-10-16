"""Cone (homogeneous) charts. Root = (a, b, s): x_a max, x_b 2nd, s = sign(x_b). Scale lam = |x_b| + x_a - x_b > 0.
Normalised coordinates w = (theta, eps_1..eps_8, 1):
  s=+ : x_a = 1,           x_b = theta,  x_j = theta - eps_j         (theta in [0,1])
  s=- : x_a = 1 - 2 theta, x_b = -theta, x_j = -theta - eps_j        (theta in [0,1])
A region = the cone over a box in (theta, eps). Its Gaussian measure is the 2-D integral
  int_0^inf dlam int_theta dtheta  lam * phi(x_a) phi(x_b) prod_j [Phi(x_b - lam eps_lo) - Phi(x_b - lam eps_hi)]
with (x_a, x_b) = lam * (chart values). Deterministic Gauss-Legendre, model-free.
"""
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss
from core import T

D = 10
EMAX = 64.0
ROOTS = [(a, b, s) for s in (1, -1) for a in range(T) for b in range(T) if a != b]


def chart(a, b, s):
    M = np.zeros((T, D))
    rest = [j for j in range(T) if j not in (a, b)]
    if s > 0:
        M[a, 9] = 1.0; M[b, 0] = 1.0
        for k, j in enumerate(rest):
            M[j, 0] = 1.0; M[j, 1 + k] = -1.0
    else:
        M[a, 9] = 1.0; M[a, 0] = -2.0; M[b, 0] = -1.0
        for k, j in enumerate(rest):
            M[j, 0] = -1.0; M[j, 1 + k] = -1.0
    return M


MS = np.array([chart(*r) for r in ROOTS])
TGT = np.array([r[1] for r in ROOTS])
SGN = np.array([r[2] for r in ROOTS])


def root_boxes():
    lo = np.array([0.0] + [0.0] * 8 + [1.0]); hi = np.array([1.0] + [EMAX] * 8 + [1.0])
    return np.repeat(lo[None], len(ROOTS), 0), np.repeat(hi[None], len(ROOTS), 0)


_LAM = leggauss(40)
_TH = {n: leggauss(n) for n in (6, 12, 24)}
LAM_MAX = 9.0


def measure(sgn, wl, wu, nth=12):
    """Gaussian measure of the cone over each box. Vectorised; FLOPs ~ 40*nth*(8*2*30+40)."""
    xl, wl_ = _LAM
    lam = 0.5 * LAM_MAX * (xl + 1)                 # (L,)
    wlam = 0.5 * LAM_MAX * wl_
    xt, wt = _TH[nth]
    tl, tu = wl[:, 0:1], wu[:, 0:1]
    th = 0.5 * (tl + tu) + 0.5 * (tu - tl) * xt[None]                 # (N,n)
    s = sgn[:, None]
    xa = np.where(s > 0, 1.0, 1.0 - 2 * th)                            # (N,n)
    xb = np.where(s > 0, th, -th)
    XA = lam[None, :, None] * xa[:, None]                              # (N,L,n)
    XB = lam[None, :, None] * xb[:, None]
    f = lam[None, :, None] * np.exp(-0.5 * (XA ** 2 + XB ** 2)) / (2 * np.pi)
    for k in range(1, 9):
        f = f * (ndtr(lam[None, :, None] * wu[:, k, None, None] - XB) - ndtr(lam[None, :, None] * wl[:, k, None, None] - XB))
    return 0.5 * (tu - tl)[:, 0] * np.einsum("nln,l,n->n".replace("nln", "nlm").replace("n->n", "m->n"), f, wlam, wt)


FLOP_MEAS = 40 * 12 * (8 * 2 * 30 + 40)

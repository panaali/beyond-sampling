"""T414 deviation D1 (secondary analysis, after the pre-registered runs): near-exact slab mass of a u-box.
For fixed (u_r, g) the density factorises over the other coordinates, so u_a and every u_l integrate to Phi
differences exactly:
  mass(box) = int_{g >= t0} int_{u_r in [r_lo, r_hi]} g phi(g u_r) phi(g (u_r + 1))
              * [Phi(g (u_r + 1 + a_hi)) - Phi(g (u_r + 1 + a_lo))] * prod_l [Phi(g (u_r - l_lo)) - Phi(g (u_r - l_hi))] du_r dg
(Jacobian g^9 times g^-8 from the 8 inner integrals). Here it is evaluated by Gauss-Legendre in u_r and in
w = g^2 s / 2 (s = the box's lower bound of |y|^2), in float. This is a kill-test estimate, not a certificate; the
rigorous version (cell-wise monotone bounds) only follows a GO."""
import math

import numpy as np
import torch as th

dt = th.float64
NP, NQ, NW, WSPAN = 32, 4, 48, 45.0                                  # u_r: 32 panels x 4 GL nodes (the peak is ~0.3 wide)
_x4, _w4 = np.polynomial.legendre.leggauss(NQ)
_xr = th.as_tensor(((np.arange(NP)[:, None] + (_x4[None] + 1) / 2) / NP * 2 - 1).ravel(), dtype=dt)   # nodes on [-1, 1]
_wr = th.as_tensor(np.tile(_w4 / NP, NP), dtype=dt)
_xw, _ww = (th.as_tensor(a, dtype=dt) for a in np.polynomial.legendre.leggauss(NW))
SQ2 = math.sqrt(2)


def Phi(z):
    return 0.5 * th.erfc(-z / SQ2)


def qmass(lo, hi, t0, m):
    """lo/hi (N, 9) u-boxes, m (N,) lower bound of |y|^2 over each box (> 0); returns (N,) mass estimates"""
    N = lo.shape[0]
    ur = (lo[:, 0:1] + hi[:, 0:1]) / 2 + (hi[:, 0:1] - lo[:, 0:1]) / 2 * _xr[None]          # (N, NR)
    wr = (hi[:, 0:1] - lo[:, 0:1]) / 2 * _wr[None]
    w0 = m * t0 ** 2 / 2                                                                    # g = sqrt(2 w / m)
    w = w0[:, None] + WSPAN / 2 * (1 + _xw[None]); ww = WSPAN / 2 * _ww[None].expand(N, NW)
    g = th.sqrt(2 * w / m[:, None]); jac = 1 / (g * m[:, None])                             # dg = dw / (g m)
    G = g[:, None, :]; R = ur[:, :, None]                                                    # (N, NR, NW)
    f = G * th.exp(-(G * R) ** 2 / 2 - (G * (R + 1)) ** 2 / 2) / (2 * math.pi)
    f = f * (Phi(G * (R + 1 + hi[:, 1, None, None])) - Phi(G * (R + 1 + lo[:, 1, None, None])))
    for i in range(2, 9):
        f = f * (Phi(G * (R - lo[:, i, None, None])) - Phi(G * (R - hi[:, i, None, None])))
    return (f * wr[:, :, None] * (ww * jac)[:, None, :]).sum((1, 2))

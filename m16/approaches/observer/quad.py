"""Deterministic quadrature over the top-3 order statistics (c = 3rd, u = b-c, g = a-b), b > 0 region."""
import numpy as np
from scipy.special import ndtr
from core import phi


def gl(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return lo + (hi - lo) * (x + 1) / 2, w * (hi - lo) / 2


def grid(nc=24, nu=24, ng=24, umax=4.0, gmax=4.0):
    """Nodes (c,u,g) and weights with density 720*phi(a)phi(b)phi(c)Phi(c)^7 (positions excluded: /720 per triple
    is applied by the caller via the uniform triple prior), restricted to b = c+u > 0.
    u and g use a squared map s->s^2*max to cluster nodes at the near-tie edge."""
    b, wb = gl(nc, 0.0, 4.5)
    su, wsu = gl(nu, 0, 1); u = umax * su ** 2; wu = wsu * 2 * umax * su
    sg, wsg = gl(ng, 0, 1); g = gmax * sg ** 2; wg = wsg * 2 * gmax * sg
    B, U, G = np.meshgrid(b, u, g, indexing="ij")
    W = wb[:, None, None] * wu[None, :, None] * wg[None, None, :]
    C = B - U; A = B + G
    dens = 720 * phi(A) * phi(B) * phi(C) * ndtr(C) ** 7
    W = W * dens
    return C.ravel(), U.ravel(), G.ravel(), W.ravel()


if __name__ == "__main__":
    for n in (16, 24, 32):
        c, u, g, w = grid(n, n, n)
        print(n, w.sum(), 1 - 11 / 1024)

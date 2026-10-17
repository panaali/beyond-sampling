"""Checks for fibre4's spherical-polygon integrals: vector-area closed form vs high-order cubature."""
import numpy as np

import fibre4 as F
from fiber3 import Ops


def vec_area(poly):
    """int_P omega dOmega = 1/2 sum_e theta_e n_e (convex spherical polygon, n_e oriented inwards)."""
    c = np.mean(poly, 0)
    A = np.zeros(3)
    for q in range(len(poly)):
        a, b = poly[q], poly[(q + 1) % len(poly)]
        x = np.cross(a, b); s = np.linalg.norm(x)
        th = np.arctan2(s, a @ b)
        A += 0.5 * th * x / s
    return A if A @ c > 0 else -A


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(5):
        c = rng.standard_normal(3); c /= np.linalg.norm(c)
        # random small convex polygon around c
        e1 = np.cross(c, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(c, e1)
        ang = np.sort(rng.uniform(0, 2 * np.pi, 5)); r = 0.3
        poly = [(c + r * (np.cos(a) * e1 + np.sin(a) * e2)) for a in ang]
        poly = [v / np.linalg.norm(v) for v in poly]
        al = rng.standard_normal(3) * 0.05
        lin = vec_area(poly) @ al
        F._S, F._T, F._WW  # noqa
        num_lin = 0.0
        # high-order reference for int alpha.omega
        x, w = np.polynomial.legendre.leggauss(30); x = 0.5 * (x + 1); w = 0.5 * w
        A, B = np.meshgrid(x, x, indexing="ij"); WW = (np.outer(w, w) * (1 - A)).ravel(); S = A.ravel(); T = ((1 - A) * B).ravel()
        ref_lin = 0.0; ref_G = 0.0
        v0 = poly[0]
        for q in range(1, len(poly) - 1):
            v1, v2 = poly[q], poly[q + 1]
            det = abs(np.dot(v0, np.cross(v1, v2)))
            W = v0 + S[:, None] * (v1 - v0) + T[:, None] * (v2 - v0); rr = np.linalg.norm(W, axis=1)
            u = W @ al / rr
            ref_lin += det * np.sum(WW * u / rr ** 3); ref_G += det * np.sum(WW * F.G(u) / rr ** 3)
        got = F.poly_int(poly, [al], [1.0], Ops()); got6 = F.poly_int(poly, [al], [1.0], Ops(), "cubature")
        print(f"linear: closed form {lin:.12f} ref {ref_lin:.12f} | G: fibre4.poly_int {got:.12f} ref {ref_G:.12f} rel {(got - ref_G) / ref_G:+.1e}; 6x6 cubature rel {(got6 - ref_G) / ref_G:+.1e}")

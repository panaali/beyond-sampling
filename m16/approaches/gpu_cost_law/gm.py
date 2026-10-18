"""T192: Grundmann-Moeller (1978) fixed cubature on the (d-1)-simplex for the ray-space cone mass
mass = Gamma(d/2) / (2 pi^(d/2)) |det R| int_{simplex} |R^T u|^-d du  (deterministic, polynomially many points:
degree 2s+1 rule on an n-simplex has sum_i C(s - i + n, n) points; n = d - 1)."""
import itertools
import math

import numpy as np


def _compositions(total, parts):
    for c in itertools.combinations(range(total + parts - 1), parts - 1):
        prev = -1; out = []
        for x in c:
            out.append(x - prev - 1); prev = x
        out.append(total + parts - 1 - prev - 1)
        yield out


def gm_rule(n, s):
    """points (barycentric, n+1 coords) and weights of the degree-(2s+1) Grundmann-Moeller rule on the standard
    n-simplex, weights summing to its volume 1/n!."""
    d = 2 * s + 1; P, W = [], []
    for i in range(s + 1):
        w = (-1) ** i * 2.0 ** (-2 * s) * (d + n - 2 * i) ** d / (math.factorial(i) * math.factorial(d + n - i))
        for b in _compositions(s - i, n + 1):
            P.append([(2 * bj + 1) / (d + n - 2 * i) for bj in b]); W.append(w)
    P = np.array(P); W = np.array(W)
    return P, W / W.sum() / math.factorial(n)


def mass_gm(R, s):
    """R (m, d, d) unit rays (rows); returns masses (m,)."""
    m, d, _ = R.shape
    P, W = gm_rule(d - 1, s)
    q = ((P @ R) ** 2).sum(-1)                       # (m, K)
    c = math.gamma(d / 2) / (2 * math.pi ** (d / 2))
    return c * np.abs(np.linalg.det(R)) * (q ** (-d / 2) @ W)

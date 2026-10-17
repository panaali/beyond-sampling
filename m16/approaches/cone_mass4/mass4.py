"""T109: cheap exact Gaussian mass of a simplicial cone in R^4 (= volume of a spherical tetrahedron / (2 pi^2)).

Method: Schlafli/Plackett differentiation along ONE correlation instead of all six.
  Let C be the correlation (Gram) matrix of the 4 unit facet normals; cone = {x : N x >= 0}; mass = P(X > 0),
  X ~ N(0, C). Vary a single entry rho = C_ij (pair (i,j), complement (k,l)) with the others fixed:
      dP/drho = phi2(0,0;rho) P(X_k, X_l > 0 | X_i = X_j = 0) = [1/4 + asin(pc(rho))/(2 pi)] / (2 pi sqrt(1-rho^2)),
  pc = partial correlation of (k,l) given (i,j), a closed-form function of rho. In phi = asin(rho):
      P(rho0) = P(rho*) + (phi0 - phi*)/(8 pi) + (1/(4 pi^2)) int_{phi*}^{phi0} asin(pc(sin phi)) dphi.
  The path ends at rho*, a root of det C(rho) (quadratic in rho). There C* is rank 3 and the cone degenerates to
  (3-D polyhedral cone with 4 facets) x line, whose mass is CLOSED FORM (spherical polygon area, Girard):
      null vector lam of C*:  all signs equal -> 0;  3:1 -> triangle of the three  1/8 + sum asin(rho*)/(4 pi);
      2:2 -> quadrilateral, adjacent pairs = the 4 cross pairs:  sum_adj asin(rho*_ab) / (4 pi).
  Near the root, asin(pc) ~ pi/2 - c sqrt(phi - phi*): the substitution phi = phi* + (phi0 - phi*) s^2 makes the
  integrand analytic in s; the 1/sqrt(1-rho^2) factor is absorbed by phi. The pair and root are chosen so that the
  other root (the next singularity) is as far as possible relative to the path length.
  Gauss-Legendre with n nodes on s in [0,1].
Op counting (as in estT/m43 Ops): +, -, *, /, sqrt, asin, sin, comparisons each count 1.
"""
import math

import numpy as np

_GL = {}


def _gl(n):
    if n not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[n] = (0.5 * (x + 1), 0.5 * w)
    return _GL[n]


class Ops:
    def __init__(self):
        self.n = 0

    def add(self, k):
        self.n += k


PAIRS = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]


def _det3(m, r, c):
    a = [[m[r[0]][c[0]], m[r[0]][c[1]], m[r[0]][c[2]]], [m[r[1]][c[0]], m[r[1]][c[1]], m[r[1]][c[2]]],
         [m[r[2]][c[0]], m[r[2]][c[1]], m[r[2]][c[2]]]]
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))                       # 14 ops


def _cof(m, r, c):
    rr = [q for q in range(4) if q != r]; cc = [q for q in range(4) if q != c]
    return (-1) ** (r + c) * _det3(m, rr, cc)


def corr_from_normals(N, ops):
    N = np.asarray(N, float)
    U = N / np.sqrt((N * N).sum(1))[:, None]; ops.add(4 * (4 + 3 + 1 + 4))
    C = U @ U.T; ops.add(6 * 7)
    return C


def mass_from_corr(C, ops=None, n=8, return_info=False):
    ops = ops if ops is not None else Ops()
    m = C.tolist()
    # adjugate entries for the 6 off-diagonal pairs + det (det from row-0 cofactors)
    cof = {}
    for (i, j, k, l) in PAIRS:
        cof[(i, j)] = _cof(m, i, j); ops.add(15)
    c00 = _cof(m, 0, 0); c11 = _cof(m, 1, 1); ops.add(30)
    det = m[0][0] * c00 + m[0][1] * cof[(0, 1)] + m[0][2] * cof[(0, 2)] + m[0][3] * cof[(0, 3)]; ops.add(7)
    if det <= 0:
        det = 0.0
    # D(rho0 + h) = det + 2 cof_ij h - (1 - rho_kl^2) h^2 ; pick pair + root with the far root farthest
    best = None
    for (i, j, k, l) in PAIRS:
        q = 1.0 - m[k][l] ** 2
        g = cof[(i, j)]
        disc = math.sqrt(max(g * g + det * q, 0.0))
        if q <= 1e-300:
            continue
        h1, h2 = (g - disc) / q, (g + disc) / q          # h1 <= 0 <= h2 roots of q h^2 - 2 g h - det = 0
        near, far = (h1, h2) if -h1 <= h2 else (h2, h1)
        score = abs(near) / max(abs(far), 1e-300)
        ops.add(14)
        if best is None or score < best[0]:
            best = (score, i, j, k, l, near)
    _, i, j, k, l, h = best
    rho0 = m[i][j]
    rs = min(1.0, max(-1.0, rho0 + h))
    # endpoint mass: C* = C with C_ij = rs; null vector from the adjugate column with largest diagonal cofactor
    ms = [row[:] for row in m]; ms[i][j] = ms[j][i] = rs
    diag = [_cof(ms, q, q) for q in range(4)]; ops.add(4 * 15)
    kk = int(np.argmax(diag)); ops.add(3)
    lam = [_cof(ms, q, kk) for q in range(4)]; ops.add(3 * 15)
    scale = max(abs(v) for v in lam); ops.add(4)
    tol = 1e-12 * scale
    pos = [q for q in range(4) if lam[q] > tol]; neg = [q for q in range(4) if lam[q] < -tol]
    zer = [q for q in range(4) if abs(lam[q]) <= tol]; ops.add(8)
    # zero components: give them the sign of the minority group (limit is continuous); tiny-measure cases
    for q in zer:
        (pos if len(pos) <= len(neg) else neg).append(q)
    if len(pos) == 0 or len(neg) == 0:
        P0 = 0.0
    elif len(pos) == 1 or len(neg) == 1:
        tri = neg if len(pos) == 1 else pos
        P0 = 0.125 + (math.asin(ms[tri[0]][tri[1]]) + math.asin(ms[tri[0]][tri[2]]) + math.asin(ms[tri[1]][tri[2]])) / (4 * math.pi)
        ops.add(10)
    else:
        P0 = sum(math.asin(max(-1.0, min(1.0, ms[a][b]))) for a in pos for b in neg) / (4 * math.pi); ops.add(10)
    # 1-D integral in phi with phi = phis + (phi0 - phis) s^2
    phi0, phis = math.asin(rho0), math.asin(rs); ops.add(2)
    dphi = phi0 - phis; ops.add(1)
    a, c, b, d, e = m[i][k], m[j][k], m[i][l], m[j][l], m[k][l]
    A0, A1, B0, B1, N0, N1 = a * a + c * c, 2 * a * c, b * b + d * d, 2 * b * d, a * b + c * d, a * d + b * c
    ops.add(14)
    s, w = _gl(n)
    acc = 0.0
    for sv, wv in zip(s, w):
        ph = phis + dphi * sv * sv
        r = math.sin(ph); c2 = 1.0 - r * r
        num = e * c2 - N0 + r * N1
        den = (c2 - A0 + r * A1) * (c2 - B0 + r * B1)
        pc = num / math.sqrt(den) if den > 0 else (1.0 if num > 0 else -1.0)
        pc = max(-1.0, min(1.0, pc))
        acc += wv * sv * math.asin(pc)
        ops.add(24)
    P = P0 + dphi / (8 * math.pi) + 2 * dphi * acc / (4 * math.pi ** 2); ops.add(8)
    if return_info:
        return P, dict(pair=(i, j), rho0=rho0, rho_star=rs, P0=P0, case=(len(pos), len(neg)))
    return P


def mass4(N, ops=None, n=8):
    """Gaussian mass of the cone {x in R^4 : N x >= 0}, N = 4 x 4 facet normals (rows)."""
    ops = ops if ops is not None else Ops()
    return mass_from_corr(corr_from_normals(N, ops), ops, n)


def mass4_rays(R, ops=None, n=8):
    """Gaussian mass of the simplicial cone spanned by the rows of R (as estT.orthant): normals = rows of inv(R^T)."""
    ops = ops if ops is not None else Ops()
    B = np.linalg.inv(np.asarray(R, float).T); ops.add(4 ** 3)
    return mass4(B, ops, n)

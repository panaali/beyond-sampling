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


def adjugate_sym(m, ops):
    """All 10 cofactors of a symmetric 4x4 matrix via shared 2x2 minors of rows (0,1) and (2,3); returns (adj, det)."""
    M01, M23 = {}, {}
    for a in range(4):
        for b in range(a + 1, 4):
            M01[(a, b)] = m[0][a] * m[1][b] - m[0][b] * m[1][a]
            M23[(a, b)] = m[2][a] * m[3][b] - m[2][b] * m[3][a]
    ops.add(36)

    def cof(r, c):
        c1, c2, c3 = [q for q in range(4) if q != c]
        if r < 2:
            rr, Mm = 1 - r, M23
        else:
            rr, Mm = 5 - r, M01
        d = m[rr][c1] * Mm[(c2, c3)] - m[rr][c2] * Mm[(c1, c3)] + m[rr][c3] * Mm[(c1, c2)]
        return d if (r + c) % 2 == 0 else -d
    adj = [[0.0] * 4 for _ in range(4)]
    for r, c in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)):
        adj[r][c] = adj[c][r] = cof(r, c)
    ops.add(50)
    det = m[0][0] * adj[0][0] + m[0][1] * adj[0][1] + m[0][2] * adj[0][2] + m[0][3] * adj[0][3]; ops.add(7)
    return adj, det


KAPPA_GOOD = 1e9          # early exit threshold for the candidate search (set by calibration; inf = search all)
TOL = 1e-9                # target quadrature error (relative to 1)
NMAX = 40


def nodes_for(kappa):
    """Gauss-Legendre nodes for a nearest singularity at distance kappa from [0,1] (in s): error ~ rhoB^(-2n),
    rhoB = Bernstein ellipse parameter; the worst case of real (beyond s=1) vs imaginary (at s=0) placement."""
    if kappa == math.inf:
        return 2
    rb = 1.0 + 2.0 * kappa + 2.0 * math.sqrt(kappa * (1.0 + kappa))     # real offset kappa beyond the end (worst)
    if rb <= 1.0 + 1e-12:
        return NMAX
    return int(min(NMAX, max(2, math.ceil(math.log(NODE_C / TOL) / (2 * math.log(rb))))))


NODE_C = 1.0


def mass_from_corr(C, ops=None, n=None, return_info=False):
    """Orthant probability P(X > 0), X ~ N(0, C), C a 4 x 4 correlation matrix. n = Gauss nodes (None: adaptive)."""
    ops = ops if ops is not None else Ops()
    m = C.tolist() if hasattr(C, "tolist") else C
    adj, det = adjugate_sym(m, ops)
    if det < 0:
        det = 0.0
    # Candidates: for each pair the nearer root of D(rho0 + h) = det + 2 adj_ij h - (1 - rho_kl^2) h^2 as endpoint.
    # Singularities of the integrand (in rho): the other D root and the roots of the two 3x3 principal minors
    # containing (i,j): rho = rho_ik rho_jk +- sqrt((1-rho_ik^2)(1-rho_jk^2)) (same for l). With tau = 1 - h_x / h_e
    # (position on the path, endpoint 0, rho0 at 1), a singularity at tau < 0 sits at s = i sqrt(-tau) and one at
    # tau > 1 at s = sqrt(tau); kappa = distance in s from [0, 1]. Pick the candidate with the largest min kappa.
    best = None
    for (i, j, k, l) in PAIRS:
        q = 1.0 - m[k][l] * m[k][l]
        if q <= 1e-300:
            continue
        g = adj[i][j]
        disc = math.sqrt(max(g * g + det * q, 0.0))
        h1, h2 = (g - disc) / q, (g + disc) / q          # h1 <= 0 <= h2
        he, hf = (h1, h2) if -h1 <= h2 else (h2, h1)
        ops.add(14)
        rho0 = m[i][j]
        a, c, b, d = m[i][k], m[j][k], m[i][l], m[j][l]
        sk = math.sqrt(max((1 - a * a) * (1 - c * c), 0.0)); sl = math.sqrt(max((1 - b * b) * (1 - d * d), 0.0))
        xs = (hf, a * c + sk - rho0, a * c - sk - rho0, b * d + sl - rho0, b * d - sl - rho0); ops.add(20)
        if he == 0.0:
            kap = math.inf
        else:
            kap = math.inf
            for hx in xs:
                tau = 1.0 - hx / he
                kx = math.sqrt(-tau) if tau < 0 else (math.sqrt(tau) - 1.0 if tau > 1 else 0.0)
                kap = min(kap, kx)
            ops.add(5 * 6)
        if best is None or kap > best[0]:
            best = (kap, i, j, k, l, he)
        if kap > KAPPA_GOOD:
            break
    score, i, j, k, l, h = best
    rho0 = m[i][j]
    rs = min(1.0, max(-1.0, rho0 + h)); ops.add(3)
    # null vector of C* = C + h (e_i e_j' + e_j e_i'):  lam ~ (h adj_ij + det) adj[:, i] - h adj_ii adj[:, j]
    al, be = h * adj[i][j] + det, -h * adj[i][i]; ops.add(3)
    lam = [al * adj[q][i] + be * adj[q][j] for q in range(4)]; ops.add(12)
    scale = max(abs(v) for v in lam); ops.add(4)
    if scale <= 1e-300:                                  # C itself singular and h = 0: null vector = an adj column
        col = i if abs(adj[i][i]) >= abs(adj[j][j]) else j
        lam = [adj[q][col] for q in range(4)]; scale = max(abs(v) for v in lam)
    tol = 1e-12 * scale
    pos = [q for q in range(4) if lam[q] > tol]; neg = [q for q in range(4) if lam[q] < -tol]
    zer = [q for q in range(4) if abs(lam[q]) <= tol]; ops.add(8)
    for q in zer:
        (pos if len(pos) <= len(neg) else neg).append(q)

    def rr(a, b):
        return rs if {a, b} == {i, j} else m[a][b]
    if len(pos) == 0 or len(neg) == 0:
        P0 = 0.0
    elif len(pos) == 1 or len(neg) == 1:
        t3 = neg if len(pos) == 1 else pos
        P0 = 0.125 + (math.asin(rr(t3[0], t3[1])) + math.asin(rr(t3[0], t3[2])) + math.asin(rr(t3[1], t3[2]))) / (4 * math.pi)
        ops.add(7)
    else:
        P0 = sum(math.asin(max(-1.0, min(1.0, rr(a, b)))) for a in pos for b in neg) / (4 * math.pi); ops.add(9)
    phi0, phis = math.asin(rho0), math.asin(rs); ops.add(2)
    dphi = phi0 - phis; ops.add(1)
    a, c, b, d, e = m[i][k], m[j][k], m[i][l], m[j][l], m[k][l]
    A0, A1, B0, B1, N0, N1 = a * a + c * c, 2 * a * c, b * b + d * d, 2 * b * d, a * b + c * d, a * d + b * c
    ops.add(14)
    nn = n if n is not None else nodes_for(score); ops.add(8)
    s, w = _gl(nn)
    acc = 0.0
    for sv, wv in zip(s, w):
        ph = phis + dphi * sv * sv
        r = math.sin(ph); c2 = 1.0 - r * r
        num = e * c2 - N0 + r * N1
        den = (c2 - A0 + r * A1) * (c2 - B0 + r * B1)
        pc = num / math.sqrt(den) if den > 0 else (1.0 if num > 0 else -1.0)
        pc = max(-1.0, min(1.0, pc))
        acc += wv * sv * math.asin(pc)
        ops.add(25)
    P = P0 + dphi / (8 * math.pi) + dphi * acc / (2 * math.pi ** 2); ops.add(6)
    if return_info:
        return P, dict(pair=(i, j), rho0=rho0, rho_star=rs, P0=P0, case=(len(pos), len(neg)), score=score, n=nn)
    return P


def mass4(N, ops=None, n=None):
    """Gaussian mass of the cone {x in R^4 : N x >= 0}, N = 4 x 4 facet normals (rows)."""
    ops = ops if ops is not None else Ops()
    return mass_from_corr(corr_from_normals(N, ops), ops, n)


FLAT_TOL = 1e-6


def mass4_rays(R, ops=None, n=None, flat_tol=FLAT_TOL, _depth=0):
    """Gaussian mass of the simplicial cone spanned by the rows of R (as estT.orthant): normals = rows of inv(R^T).
    Flat cones (rays nearly in a hyperplane; facet normals nearly +-parallel, correlations within flat_tol of +-1 for
    >= 3 pairs) lose all their information in a float correlation matrix. They are split exactly by the signed
    decomposition from an apex w = the first unit facet normal (perpendicular to the near-hyperplane):
        1_S = sum_i eps_i 1_cone(w, F_i),  eps_i = sign(n_i . w),  F_i = facet opposite ray i,
    whose four pieces are well conditioned (w is far from every facet plane)."""
    ops = ops if ops is not None else Ops()
    R = np.asarray(R, float)
    B = np.linalg.inv(R.T); ops.add(4 ** 3)
    C = corr_from_normals(B, ops)
    near = int((np.abs(C[np.triu_indices(4, 1)]) > 1 - flat_tol).sum()); ops.add(12)
    if near < 3 or _depth > 0:
        return mass_from_corr(C, ops, n)
    U = B / np.sqrt((B * B).sum(1))[:, None]; ops.add(48)
    w = U[0]
    tot = 0.0
    for i in range(4):
        e = float(U[i] @ w); ops.add(8)
        if abs(e) < 1e-300:
            continue
        Ri = R.copy(); Ri[i] = w
        tot += math.copysign(1.0, e) * mass4_rays(Ri, ops, n, flat_tol, _depth + 1)
    return tot

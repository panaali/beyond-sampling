"""High-precision references for the Gaussian orthant probability P(X > 0), X ~ N(0, C), C 4 x 4 correlation.
ref_plackett: the ALL-correlation Plackett path (independent of mass4's single-edge path), integrated with mpmath
tanh-sinh at 30 digits: P = 1/16 + sum asin(rho_ij)/(8 pi) + (1/4pi^2) sum_ij int_0^1 rho_ij asin(pc_kl.ij(t C))
/ sqrt(1 - t^2 rho_ij^2) dt (partial correlations from the exact 4x4 inverse at each t).
ref_edge: mass4's single-edge formula, but integrated by mpmath (checks the quadrature, not the formula)."""
import mpmath as mp

mp.mp.dps = 30
PAIRS = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]


def corr_mp(N):
    """30-digit correlation matrix of the (exact float) normals N: exactly PSD, unlike a float C."""
    U = [[N[a][b] if isinstance(N[a][b], mp.mpf) else mp.mpf(float(N[a][b])) for b in range(4)] for a in range(4)]
    nr = [mp.sqrt(sum(u * u for u in row)) for row in U]
    return mp.matrix([[sum(U[a][q] * U[b][q] for q in range(4)) / (nr[a] * nr[b]) for b in range(4)] for a in range(4)])


def normals_mp(R):
    """30-digit facet normals (rows of inv(R^T)) of the cone spanned by the float rays R."""
    Rm = mp.matrix([[mp.mpf(float(R[a][b])) for b in range(4)] for a in range(4)])
    B = (Rm.T) ** -1
    return [[B[a, b] for b in range(4)] for a in range(4)]


def ref_rays(R):
    return ref_plackett(N=normals_mp(R))


def ref_plackett(N=None, C=None):
    Cm = corr_mp(N) if N is not None else mp.matrix([[mp.mpf(float(C[a][b])) for b in range(4)] for a in range(4)])

    def f(t):
        Rt = mp.matrix(4, 4)
        for a in range(4):
            for b in range(4):
                Rt[a, b] = 1 if a == b else t * Cm[a, b]
        Om = Rt ** -1
        s = 0
        for i, j, k, l in PAIRS:
            pc = mp.re(-Om[k, l] / mp.sqrt(Om[k, k] * Om[l, l]))
            pc = max(mp.mpf(-1), min(mp.mpf(1), pc))
            s += Cm[i, j] * mp.asin(pc) / mp.sqrt(max(1 - (t * Cm[i, j]) ** 2, mp.mpf(10) ** -60))
        return s
    base = mp.mpf(1) / 16 + sum(mp.asin(Cm[i, j]) for i, j, _, _ in PAIRS) / (8 * mp.pi)
    return base + mp.quad(f, [0, mp.mpf(1) / 2, 1 - mp.mpf(10) ** -3, 1]) / (4 * mp.pi ** 2)

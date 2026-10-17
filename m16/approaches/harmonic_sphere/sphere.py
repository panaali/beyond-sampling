"""Model-free harmonic analysis on S^{n-1} (all of it is PRECOMPUTE: it does not depend on the model).

- cubature(n, q): product Gauss rule on S^{n-1}, exact for every polynomial of degree <= q.
- basis(n, D, Y): orthonormal spherical harmonics of degree <= D (Dai & Xu recursive Gegenbauer basis),
  evaluated at nodes Y; columns grouped by degree.
- halfspace_coeffs(n, K): Funk-Hecke coefficients lambda_d of the indicator 1[t >= 0], so that the degree<=K
  harmonic projection of y -> 1[v.y >= 0] is  sum_d lambda_d Z_d(v.y),  Z_d(t) = (d+a)/a C_d^a(t), a=(n-2)/2.
The measure is the normalised uniform measure sigma on S^{n-1} (total mass 1).
"""
import numpy as np
from scipy.special import roots_gegenbauer, eval_gegenbauer, roots_legendre, betaln


def _gauss_t(k, npts):
    """Nodes/weights on [-1,1] for weight (1-t^2)^((k-2)/2) (the x_{k+1} marginal on S^k), normalised to sum 1."""
    if k == 2:                                      # weight 1: Legendre
        t, w = roots_legendre(npts)
    else:
        t, w = roots_gegenbauer(npts, (k - 1) / 2)
    return t, w / w.sum()


def cubature(n, q):
    """Returns (Y (N,n), w (N,), coords) exact for degree <= q. coords = list of per-level arrays used by basis():
    coords[0] = phi (angle on S^1), coords[k-1] = t_k = x_{k+1} for S^k, k>=2."""
    nphi = q + 1
    phi = 2 * np.pi * np.arange(nphi) / nphi
    Y = np.stack([np.cos(phi), np.sin(phi)], 1); w = np.full(nphi, 1 / nphi); co = [phi[:, None]]
    for k in range(2, n):                           # lift S^{k-1} -> S^k
        t, wt = _gauss_t(k, q // 2 + 1)
        N0 = len(Y)
        s = np.sqrt(1 - t ** 2)
        Y = np.concatenate([np.repeat(s[:, None], N0, 0).reshape(-1, 1) * np.tile(Y, (len(t), 1)),
                            np.repeat(t, N0)[:, None]], 1)
        w = (wt[:, None] * w[None]).ravel()
        co = [np.tile(c, (len(t), 1)) for c in co] + [np.repeat(t, N0)[:, None]]
    return Y, w, [c[:, 0] for c in co]


def _harm(n, D, co):
    """Unnormalised orthogonal harmonics on S^{n-1} up to degree D at nodes with coordinates co.
    Returns list over degree d of arrays (N, dim H_d)."""
    phi = co[0]
    H = [[np.ones_like(phi)[:, None]]]              # S^1, degree 0
    for d in range(1, D + 1):
        H.append([np.stack([np.cos(d * phi), np.sin(d * phi)], 1)])
    H = [h[0] for h in H]
    for k in range(2, n):                           # S^{k-1} harmonics H -> S^k harmonics
        t = co[k - 1]; s2 = 1 - t ** 2
        new = []
        for d in range(D + 1):
            cols = []
            for j in range(d + 1):
                a = j + (k - 1) / 2
                g = eval_gegenbauer(d - j, a, t) * s2 ** (j / 2)
                cols.append(g[:, None] * H[j])
            new.append(np.concatenate(cols, 1))
        H = new
    return H


def basis(n, D, Y, w, co):
    """Orthonormal (w.r.t. sigma) harmonic basis up to degree D at the nodes: Phi (N, m), deg (m,)."""
    H = _harm(n, D, co)
    Phi = np.concatenate(H, 1)
    deg = np.concatenate([np.full(h.shape[1], d) for d, h in enumerate(H)])
    nrm = np.sqrt((w[:, None] * Phi ** 2).sum(0))
    return Phi / nrm, deg, nrm


def coords(X):
    """Recursive spherical coordinates (as in cubature()) of unit vectors X (B,n)."""
    n = X.shape[1]; co = [None] * (n - 1); Z = X.copy()
    for k in range(n - 1, 1, -1):                   # peel x_{k+1}
        t = np.clip(Z[:, k], -1, 1); co[k - 1] = t
        r = np.linalg.norm(Z[:, :k], axis=1); r[r == 0] = 1
        Z = Z[:, :k] / r[:, None]
    co[0] = np.arctan2(Z[:, 1], Z[:, 0])
    return co


def halfspace_coeffs(n, K, npts=2000):
    """lambda_d, d=0..K, for f(t)=1[t>=0] on S^{n-1}: proj_d f(v.) = lambda_d Z_d(v.)."""
    a = (n - 2) / 2
    th, wt = roots_legendre(npts); th = (th + 1) * np.pi / 4; wt = wt * np.pi / 4     # theta in [0, pi/2]
    t = np.sin(th); dens = np.cos(th) ** (2 * a)     # dt (1-t^2)^{a-1/2} = cos^{2a} theta dtheta
    Z = np.exp(betaln(0.5, a + 0.5))                 # int_{-1}^1 (1-t^2)^{a-1/2} dt
    lam = np.array([(wt * dens * eval_gegenbauer(d, a, t) / eval_gegenbauer(d, a, 1.0)).sum() / Z
                    for d in range(K + 1)])
    lam[2::2] = 0.0                                  # exact zeros (odd part of 1[t>=0] - 1/2)
    return lam


def zonal(n, K, lam, tv):
    """sum_{d<=K} lam_d Z_d(t) at t (any shape), via the Gegenbauer three-term recurrence. ~5K flops per point."""
    a = (n - 2) / 2
    c0 = np.ones_like(tv); c1 = 2 * a * tv
    out = lam[0] * c0 + lam[1] * (1 + a) / a * c1
    for d in range(2, K + 1):
        c0, c1 = c1, (2 * tv * (d + a - 1) * c1 - (d + 2 * a - 2) * c0) / d
        if lam[d] != 0.0:
            out = out + lam[d] * (d + a) / a * c1
    return out


class Space:
    """Degree-<=D harmonic space on S^{n-1} with an exact cubature for degree 4D (needed for exact cuts)."""

    def __init__(self, n, D):
        self.n, self.D = n, D
        self.Y, self.w, co = cubature(n, 4 * D)
        self.Phi, self.deg, self.nrm = basis(n, D, self.Y, self.w, co)
        self.lam = halfspace_coeffs(n, 2 * D)
        self.N, self.m = self.Phi.shape
        self.PhiTw = (self.Phi * self.w[:, None]).T.copy()
        self.flops = 0.0                             # model-dependent work done in this space

    def precompute_flops(self):
        return float(self.N * self.m * 40)           # rough: building the basis and rules

    def cut(self, C, V):
        """C (m,B) coefficient columns, V (B,n) unit normals. Returns coeffs of proj_D(g_b * 1[v_b.y >= 0])."""
        B = C.shape[1]
        ch = max(1, int(2e7 // (8 * self.N)))                         # chunk columns: <~200 MB of temporaries
        if B > ch:
            out = np.concatenate([self.cut(C[:, k:k + ch], V[k:k + ch]) for k in range(0, B, ch)], 1)
            return out
        F = self.Phi @ C                                              # N x B
        S = zonal(self.n, 2 * self.D, self.lam, self.Y @ V.T)         # N x B
        out = self.PhiTw @ (F * S)
        self.flops += B * (4.0 * self.N * self.m + self.N * (2 * self.n + 6 * self.D) + self.N)
        return out

    def at(self, X):
        """Basis functions at unit vectors X (B,n) -> (B,m)."""
        return np.concatenate(_harm(self.n, self.D, coords(X)), 1) / self.nrm

    def mass_in(self, C, V):
        """int g_b 1[v_b.y >= 0] dsigma for coefficient columns C (m,B), normals V (B,n). Funk-Hecke: O(m) each."""
        P = self.at(V)                                                # B x m
        lamdeg = self.lam[self.deg]
        self.flops += C.shape[1] * (3.0 * self.m + 10 * self.m)       # basis evaluation ~10 flops/function
        return np.einsum("bm,mb->b", P * lamdeg, C)

    def one(self):
        c = np.zeros(self.m); c[0] = 1.0             # g = 1 (deg-0 basis function is the constant 1)
        return c

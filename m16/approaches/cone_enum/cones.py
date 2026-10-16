"""Error-cone enumeration + model-free Gaussian-mass integration for M_16,10.

A cone is keyed by (full 10x16 gate pattern, predicted class k, position of max i, position of 2nd j).
On it, everything is linear: constraints A x >= 0 with rows
  gate rows   s_{t,n} * M_t[n]      (pre-activation sign at step t, neuron n)
  logit rows  (L_k - L_l)           l != k
  order rows  e_i - e_j, e_j - e_l  l != i,j
Error cone iff k != j.  Mass = P_{x~N(0,I)}(A x >= 0) via 1-D preintegration along the Chebyshev
direction c + scrambled Sobol over the 9-dim complement (a Genz-style SOV integrator; model-free).
Compute is counted in FLOPs (1 forward pass = FWD flops).
"""
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.stats import qmc, norm

T, H = 10, 16
FWD = 10 * (16 * 16 + 16) * 2 + 10 * 16 * 2   # ~5.8k flops


class Counter:
    def __init__(self): self.flops = 0; self.lps = 0; self.fwd = 0; self.cones = 0
    @property
    def fwd_equiv(self): return self.flops / FWD


def forward_point(model, x):
    Wi, Wh, Wo = model
    h = np.zeros(H)
    for t in range(T):
        h = np.maximum(Wh @ h + Wi * x[t], 0)
    return int(np.argmax(Wo @ h))


def cone_of(model, x, C):
    """Key and constraint matrix of the cone containing x (linear maps tracked; ~10 fwd passes of flops)."""
    Wi, Wh, Wo = model
    M = np.zeros((H, T)); rows = []; gates = []
    for t in range(T):
        P = Wh @ M; P[:, t] += Wi            # pre-activation map at step t
        pre = P @ x
        g = pre > 0
        gates.append(g)
        rows.append(np.where(g[:, None], P, -P))
        M = P * g[:, None]
    C.flops += T * (2 * H * H * T + 2 * H * T)
    L = Wo @ M
    k = int(np.argmax(L @ x))
    o = np.argsort(-x); i, j = int(o[0]), int(o[1])
    lr = L[k][None] - np.delete(L, k, 0)
    E = np.eye(T)
    orr = [E[i] - E[j]] + [E[j] - E[l] for l in range(T) if l not in (i, j)]
    A = np.vstack(rows + [lr, np.array(orr)])
    nrm = np.linalg.norm(A, axis=1)
    A = A[nrm > 1e-12] / nrm[nrm > 1e-12, None]
    key = (np.packbits(np.concatenate(gates)).tobytes(), k, i, j)
    return key, A


def cheb(A, C):
    """Chebyshev-like interior direction: max r s.t. A x >= r, |x|_inf <= 1."""
    m, d = A.shape
    res = linprog(np.r_[np.zeros(d), -1.0], A_ub=np.c_[-A, np.ones(m)], b_ub=np.zeros(m),
                  bounds=[(-1, 1)] * d + [(0, 1)], method="highs")
    C.lps += 1
    it = max(getattr(res, "nit", 0) or 0, d)
    C.flops += 2 * (m + d) * (d + 1) * it * 4          # rough simplex cost: 4 dense ops per pivot row
    if res.status != 0 or res.x[-1] < 1e-10:
        return None
    c = res.x[:-1]; nc = np.linalg.norm(c); return c / nc, res.x[-1] / nc


_SOB = {}
def sobol_normals(n, d, seed):
    key = (n, d, seed)
    if key not in _SOB:
        u = qmc.Sobol(d, scramble=True, seed=seed).random(n)
        _SOB[key] = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return _SOB[key]


def cone_mass(A, c, C, n=128, seed=0):
    """P(A x >= 0), x ~ N(0,I_10).  x = s c + Q z, s ~ N(0,1): each row gives s >= -a.Qz / a.c.
    Returns mass, per-point (L, z, binding row) for facet discovery."""
    d = A.shape[1]
    Q = np.linalg.svd(np.eye(d) - np.outer(c, c))[0][:, :d - 1]
    Z = sobol_normals(n, d - 1, seed)
    Y = Z @ Q.T                                   # n x d
    ac = A @ c
    R = -(Y @ A.T) / ac[None]                     # n x m
    b = R.argmax(1); Lb = R[np.arange(n), b]
    C.flops += 2 * n * d * (d - 1) + 2 * n * A.shape[0] * d + 3 * n * A.shape[0]
    w = norm.cdf(-Lb)
    return w.mean(), Lb, Y, b, w


def cone_mass_is(A, c, rho, C, n=128, stages=4, seed=0):
    """Preintegration along c + cross-entropy importance sampling over the 9-dim complement.
    Integrand f(z) = phi_9(z) Phi(-L(z)); proposal N(mu, S) adapted by weighted moments (model-free).
    rho = inscribed angular radius (from the Chebyshev LP) sets the initial scale."""
    d = A.shape[1]
    Q = np.linalg.svd(np.eye(d) - np.outer(c, c))[0][:, :d - 1]
    AQ = A @ Q; ac = A @ c
    C.flops += 2 * A.shape[0] * d * (d - 1)
    mu = np.zeros(d - 1); S = np.eye(d - 1)
    for st in range(stages):
        Z0 = sobol_normals(n, d - 1, seed * 31 + st)
        Lc = np.linalg.cholesky(S + 1e-14 * np.eye(d - 1))
        Z = mu + Z0 @ Lc.T
        R = -(Z @ AQ.T) / ac[None]
        b = R.argmax(1); Lb = R[np.arange(n), b]
        logq = -0.5 * (Z0 ** 2).sum(1) - np.log(np.diag(Lc)).sum()
        logp = -0.5 * (Z ** 2).sum(1)
        lw = logp - logq + norm.logcdf(-Lb)
        C.flops += 2 * n * A.shape[0] * (d - 1) + 4 * n * A.shape[0] + 4 * n * (d - 1) ** 2 + 60 * n
        m = lw.max(); w = np.exp(lw - m)
        est = np.exp(m) * w.mean()
        if st < stages - 1:
            wn = w / w.sum()
            mu = wn @ Z
            Dz = Z - mu
            S = 1.3 * (Dz * wn[:, None]).T @ Dz + 1e-12 * np.eye(d - 1)
    return est, Lb, Z @ Q.T, b, np.exp(lw)


from scipy.special import ndtr, ndtri, log_ndtr


def _trunc_std(lo, hi, u):
    """Sample N(0,1) truncated to [lo,hi] by inverse cdf, vectorised, using the tail on the near side."""
    flip = lo > 0                       # work in the lower tail for accuracy
    a = np.where(flip, -hi, lo); b = np.where(flip, -lo, hi)
    Fa, Fb = ndtr(a), ndtr(b)
    z = ndtri(np.clip(Fa + u * (Fb - Fa), 1e-300, 1 - 1e-16))
    z = np.clip(z, a, b)
    return np.where(flip, -z, z)


def cone_mass_mcmc(A, c, C, K=32, steps=12, n=256, seed=0, infl=1.5):
    """(1) Gaussian hit-and-run inside the cone (exact 1-D truncated-normal moves; model-free) to learn
    the cone's shape; (2) preintegrate along the samples' mean direction and importance-sample the
    9-dim complement with a Gaussian fitted to the chain (inflated), scrambled-Sobol driven.
    Unbiased given the proposal; returns (mass, rel. s.e., binding-row stats)."""
    rng = np.random.default_rng(seed)
    m, d = A.shape
    X = np.tile(c * 2.0, (K, 1))
    keep = []
    for s in range(steps):
        D = rng.standard_normal((K, d)); D /= np.linalg.norm(D, axis=1, keepdims=True)
        ax = X @ A.T; ad = D @ A.T                          # K x m
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -ax / ad
        lo = np.where(ad > 0, r, -np.inf).max(1); hi = np.where(ad < 0, r, np.inf).min(1)
        mu = -(X * D).sum(1)
        t = mu + _trunc_std(lo - mu, hi - mu, rng.random(K))
        X = X + t[:, None] * D
        if s >= 3: keep.append(X.copy())
        C.flops += 4 * K * m * d + 60 * K
    S = np.vstack(keep)
    cm = S.mean(0); cm /= np.linalg.norm(cm)
    ac = A @ cm
    if ac.min() <= 0: cm = c; ac = A @ cm
    Q = np.linalg.svd(np.eye(d) - np.outer(cm, cm))[0][:, :d - 1]
    Yz = (S - np.outer(S @ cm, cm)) @ Q
    # condition on s: the y-marginal of the cone ∝ phi(y) Phi(-L(y)); fit the chain's y
    mu = Yz.mean(0); Sig = np.cov(Yz.T) * infl + 1e-10 * np.eye(d - 1)
    Lc = np.linalg.cholesky(Sig)
    Z0 = sobol_normals(n, d - 1, seed)
    Z = mu + Z0 @ Lc.T
    AQ = A @ Q
    R = -(Z @ AQ.T) / ac[None]
    b = R.argmax(1); Lb = R[np.arange(n), b]
    lw = -0.5 * (Z ** 2).sum(1) + 0.5 * (Z0 ** 2).sum(1) + np.log(np.diag(Lc)).sum() + log_ndtr(-Lb)
    C.flops += 2 * m * d * d + 2 * n * m * (d - 1) + 4 * n * m + 4 * n * (d - 1) ** 2 + 60 * n
    w = np.exp(lw)
    est = w.mean(); rse = w.std() / np.sqrt(n) / max(est, 1e-300)
    return est, rse, (Z @ Q.T, Lb, b, w, cm)

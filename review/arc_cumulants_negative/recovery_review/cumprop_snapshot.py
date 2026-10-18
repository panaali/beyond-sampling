"""Independent (T80) moment-propagation primitives for ReLU RNNs. Written from scratch; does not import arc_kprop.

Two closures of one ReLU layer acting on a Gaussian-summarised pre-activation u ~ (mu, C):
  * arc2  : ARC's K=2 rule (arXiv:2605.05179, Algorithm 2 as described there): exact 1-D mean and variance per
            unit, off-diagonal covariance by the leading Hermite term C_ij c_i c_j, c_i = E[relu'(u_i)].
  * gauss : the full Gaussian-closure answer (all Hermite orders for each pair = Wright et al. series): exact
            E[relu(u_i) relu(u_j)] for a bivariate normal, by a 1-D conditional integral (Gauss-Legendre, split
            at the kink), so it is exact also for correlation +-1.
Linear pass-through coordinates y (e.g. stored inputs x) get Cov(relu(u_i), y) = c_i Cov(u_i, y) (Stein; exact for
jointly Gaussian (u, y), and it is also what the leading Hermite term gives).
"""
import numpy as np
from scipy.special import ndtr

SQ2PI = np.sqrt(2 * np.pi)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


def npdf(z):
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-0.5 * np.minimum(z * z, 1e6)) / SQ2PI


def relu1(mu, var):
    """E relu(U), Var relu(U), Phi(mu/sd) for U ~ N(mu, var) (elementwise)."""
    sd = np.sqrt(np.maximum(var, 1e-300))
    a = mu / sd
    P, p = ndtr(a), npdf(a)
    m = mu * P + sd * p
    e2 = (mu * mu + var) * P + mu * sd * p
    return m, np.maximum(e2 - m * m, 0.0), P


def _erelu(m, s):
    s = np.maximum(s, 1e-300)
    return m * ndtr(m / s) + s * npdf(m / s)


def _gl(lo, hi):
    """Gauss-Legendre nodes/weights on [lo, hi] (broadcast arrays)."""
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    return mid[..., None] + half[..., None] * _GL_X, half[..., None] * _GL_W


def relu_pair_moment(mi, si, mj, sj, rho):
    """E[relu(U) relu(V)], (U, V) bivariate normal with means mi, mj, sds si, sj, correlation rho (arrays).
    U = mi + si Z; V | Z ~ N(mj + rho sj Z, sj^2 (1-rho^2)). Integrate over Z > -mi/si, split at the kink of V."""
    a = -mi / si
    b = np.maximum(a, 0.0) + 9.0
    sv = sj * np.sqrt(np.maximum(1 - rho * rho, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        zk = np.where(np.abs(rho) > 1e-12, -mj / (rho * sj), np.inf)
    zk = np.clip(zk, a, b)
    tot = 0.0
    for lo, hi in ((a, zk), (zk, b)):
        z, w = _gl(lo, hi)
        f = npdf(z) * (mi[..., None] + si[..., None] * z) * _erelu(mj[..., None] + (rho * sj)[..., None] * z,
                                                                   sv[..., None])
        tot = tot + (w * f).sum(-1)
    return tot


def relu_layer(mu_u, C_u, mode):
    """Moments of relu(u) for u summarised by (mu_u (n,), C_u (n,n)). Returns mean, cov, c (=Phi(alpha))."""
    var = np.maximum(np.diag(C_u).copy(), 1e-300)
    m, v, c = relu1(mu_u, var)
    if mode == "arc2":
        cov = C_u * np.outer(c, c)
    elif mode == "gauss":
        n = len(mu_u)
        sd = np.sqrt(var)
        iu = np.triu_indices(n, 1)
        rho = np.clip(C_u[iu] / (sd[iu[0]] * sd[iu[1]]), -1.0, 1.0)
        E = relu_pair_moment(mu_u[iu[0]], sd[iu[0]], mu_u[iu[1]], sd[iu[1]], rho)
        cov = np.zeros((n, n))
        cov[iu] = E - m[iu[0]] * m[iu[1]]
        cov = cov + cov.T
        # units with (numerically) zero variance are constants 0 or mu: covariance 0
        dead = var < 1e-24
        cov[dead, :] = 0; cov[:, dead] = 0
    else:
        raise ValueError(mode)
    np.fill_diagonal(cov, v)
    return m, cov, c


def rnn_step_joint(mu, C, Wh, Wi, t, H, mode):
    """State s = (h (H), x (T)) summarised by mean/cov. One step h <- relu(Wh h + Wi x_t), x unchanged."""
    D = len(mu)
    A = np.zeros((H, D)); A[:, :H] = Wh; A[:, H + t] = Wi
    mu_u = A @ mu
    C_uu = A @ C @ A.T
    C_ux = A @ C[:, H:]                      # Cov(u, x)
    m, cov, c = relu_layer(mu_u, C_uu, mode)
    mu2 = mu.copy(); mu2[:H] = m
    C2 = C.copy()
    C2[:H, :H] = cov
    C2[:H, H:] = c[:, None] * C_ux
    C2[H:, :H] = C2[:H, H:].T
    return mu2, C2


def rnn_step_inject(mu, C, Wh, Q, mode):
    """Random-RNN step with a fresh input independent of h: u = Wh h + (input with covariance Q)."""
    return relu_layer(Wh @ mu, Wh @ C @ Wh.T + Q, mode)[:2]


def relu_layer_b(mu_u, C_u, mode):
    """Batched relu_layer: mu_u (B, n), C_u (B, n, n). Returns mean (B, n), cov (B, n, n), c (B, n)."""
    B, n = mu_u.shape
    var = np.maximum(np.einsum("bii->bi", C_u), 1e-300)
    m, v, c = relu1(mu_u, var)
    if mode == "arc2":
        cov = C_u * c[:, :, None] * c[:, None, :]
    elif mode == "gauss":
        sd = np.sqrt(var)
        i0, i1 = np.triu_indices(n, 1)
        rho = np.clip(C_u[:, i0, i1] / (sd[:, i0] * sd[:, i1]), -1.0, 1.0)
        E = relu_pair_moment(mu_u[:, i0], sd[:, i0], mu_u[:, i1], sd[:, i1], rho)
        cov = np.zeros((B, n, n))
        cov[:, i0, i1] = E - m[:, i0] * m[:, i1]
        cov = cov + cov.transpose(0, 2, 1)
        dead = var < 1e-24
        cov = np.where(dead[:, :, None] | dead[:, None, :], 0.0, cov)
    else:
        raise ValueError(mode)
    d = np.arange(n)
    cov[:, d, d] = v
    return m, cov, c


def gh_nodes(n):
    """Nodes/weights for N(0,1). n > 0: probabilists' Gauss-Hermite. n < 0: |n|/2 Gauss-Legendre nodes on each
    half-line [0, 8] x phi (the 'split' rule: exact for functions linear on each half-line, i.e. for h_1)."""
    if n > 0:
        z, w = np.polynomial.hermite_e.hermegauss(n)
        return z, w / w.sum()
    k = -n // 2
    g, gw = np.polynomial.legendre.leggauss(k)
    z = 4 * (g + 1); w = 4 * gw * npdf(z)
    z = np.concatenate([-z[::-1], z]); w = np.concatenate([w[::-1], w])
    return z, w / w.sum()


def rim_hidden(Wi, Wh, T, m, n, mode, record=None):
    """'Recent-input mixture' (T80 variant a, structure vs randomness): the last m inputs are integrated exactly
    on an n-node Gauss-Hermite grid (each grid point = one mixture component, the exactly known x-driven part),
    the older inputs are summarised by a Gaussian/ARC-K2 residual (mean, cov) per component. Each step adds the
    new input as a grid dimension and merges the oldest one by moment matching (law of total covariance).
    m = 0: x_t-conditioned closure; m >= T: plain tensor-product quadrature (no closure error).
    Returns overall (mean, cov) of h_t after each step in `record` (list) and the final components."""
    H = Wh.shape[0]
    z, w = gh_nodes(n)
    W = np.ones(1); MU = np.zeros((1, H)); C = np.zeros((1, H, H))
    keys = [()]
    for t in range(T):
        B = len(W)
        n = len(z)
        mu_u = (MU @ Wh.T)[:, None, :] + z[None, :, None] * Wi[None, None, :]      # (B, n, H)
        Cu = np.einsum("ij,bjk,lk->bil", Wh, C, Wh)                              # (B, H, H)
        Cu = np.broadcast_to(Cu[:, None], (B, n, H, H)).reshape(B * n, H, H)
        mu_f = mu_u.reshape(B * n, H)
        if not C.any():                          # deterministic components (all inputs on the grid so far)
            mm, cc = np.maximum(mu_f, 0), np.zeros_like(Cu)
        else:
            parts = [relu_layer_b(mu_f[i:i + 1024], Cu[i:i + 1024], mode) for i in range(0, B * n, 1024)]
            mm = np.concatenate([p[0] for p in parts]); cc = np.concatenate([p[1] for p in parts])
        Wn = (W[:, None] * w[None, :]).reshape(-1)
        kn = [k + (j,) for k in keys for j in range(n)]
        if len(kn[0]) > m:                       # merge over the oldest grid dimension
            kn = [k[1:] for k in kn]
            uniq = {}
            for i, k in enumerate(kn):
                uniq.setdefault(k, []).append(i)
            keys = list(uniq)
            W2 = np.empty(len(keys)); MU2 = np.empty((len(keys), H)); C2 = np.empty((len(keys), H, H))
            for a, k in enumerate(keys):
                ii = uniq[k]; ww = Wn[ii]; s = ww.sum()
                mu_ = (ww[:, None] * mm[ii]).sum(0) / s
                dm = mm[ii] - mu_
                C2[a] = (ww[:, None, None] * (cc[ii] + dm[:, :, None] * dm[:, None, :])).sum(0) / s
                W2[a] = s; MU2[a] = mu_
            W, MU, C = W2, MU2, C2
        else:
            keys, W, MU, C = kn, Wn, mm, cc
        if record is not None:
            mu_ = (W[:, None] * MU).sum(0)
            dm = MU - mu_
            record.append((mu_, (W[:, None, None] * (C + dm[:, :, None] * dm[:, None, :])).sum(0)))
    return W, MU, C

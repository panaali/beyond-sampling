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

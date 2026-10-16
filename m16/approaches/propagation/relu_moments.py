"""Exact first and second moments of ReLU applied to a jointly Gaussian vector (no sampling)."""
import numpy as np
from scipy.special import ndtr, owens_t

SQ2PI = np.sqrt(2 * np.pi)


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def Phi2(h, k, r):
    """Standard bivariate normal CDF P(U<h, V<k), corr r, via Owen's T (vectorised, deterministic)."""
    h = np.where(np.abs(h) < 1e-10, 1e-10, h)
    k = np.where(np.abs(k) < 1e-10, 1e-10, k)
    s = np.sqrt(1 - r * r)
    beta = np.where(h * k > 0, 0.0, 0.5)
    return 0.5 * ndtr(h) + 0.5 * ndtr(k) - owens_t(h, (k - r * h) / (h * s)) - owens_t(k, (h - r * k) / (k * s)) - beta


def relu_moments(mu, C):
    """mu (B,n), C (B,n,n) for pre-activations u. Returns E[relu(u)] (B,n), Cov[relu(u)] (B,n,n),
    and Phi(alpha) (B,n) (Stein factor: Cov(relu(u_i), y) = Phi(alpha_i) Cov(u_i, y) for any y jointly
    Gaussian with u)."""
    var = np.maximum(np.einsum('bii->bi', C), 1e-24)
    sd = np.sqrt(var)
    a = mu / sd
    Pa = ndtr(a); pa = phi(a)
    m = mu * Pa + sd * pa
    E2 = (mu * mu + var) * Pa + mu * sd * pa
    # pairwise
    r = np.clip(C / (sd[:, :, None] * sd[:, None, :]), -0.999999, 0.999999)
    ai, aj = a[:, :, None], a[:, None, :]
    s = np.sqrt(1 - r * r)
    P = Phi2(np.broadcast_to(ai, r.shape), np.broadcast_to(aj, r.shape), r)
    Q1 = ndtr((aj - r * ai) / s)      # Phi((b - r a)/s)
    Q2 = ndtr((ai - r * aj) / s)
    pai, paj = pa[:, :, None], pa[:, None, :]
    EU = -pai * Q1 - r * paj * Q2
    EV = -paj * Q2 - r * pai * Q1
    phi2 = np.exp(-(ai * ai - 2 * r * ai * aj + aj * aj) / (2 * s * s)) / (2 * np.pi * s)
    EUV = r * P - r * ai * pai * Q1 - r * aj * paj * Q2 + s * s * phi2
    mi, mj = mu[:, :, None], mu[:, None, :]
    si, sj = sd[:, :, None], sd[:, None, :]
    EXY = mi * mj * P - mi * sj * EV - mj * si * EU + si * sj * EUV
    cov = EXY - m[:, :, None] * m[:, None, :]
    n = mu.shape[1]
    d = np.arange(n)
    cov[:, d, d] = E2 - m * m
    return m, cov, Pa

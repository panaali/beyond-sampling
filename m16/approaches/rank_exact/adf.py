"""Assumed-density (Gaussian) moment propagation of the RNN given an input mean/covariance. Model calls: none
beyond weight algebra; cost is counted in FLOPs (~ATT forward-pass equivalents)."""
import numpy as np
from scipy.special import owens_t, ndtr
from scipy.stats import norm
T, H = 10, 16
phi = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)

def Phi2(h, k, r):
    """Bivariate normal CDF P(X<h, Y<k), corr r (vectorised, Owen's T)."""
    h = np.asarray(h, float); k = np.asarray(k, float); r = np.clip(r, -0.999999, 0.999999)
    s = np.sqrt(1 - r * r)
    hs = np.where(np.abs(h) < 1e-12, 1e-12, h); ks = np.where(np.abs(k) < 1e-12, 1e-12, k)
    a1 = (ks - r * hs) / (hs * s); a2 = (hs - r * ks) / (ks * s)
    beta = np.where(hs * ks > 0, 0.0, np.where(hs * ks < 0, 0.5, 0.0))
    return 0.5 * ndtr(hs) + 0.5 * ndtr(ks) - owens_t(hs, a1) - owens_t(ks, a2) - beta * 0 - np.where(hs * ks < 0, 0.5, 0.0)

def relu_moments(mu, C):
    """Exact mean and covariance of relu(u) for u ~ N(mu, C)."""
    s = np.sqrt(np.maximum(np.diag(C), 1e-18)); a = mu / s
    Pa = ndtr(a); pa = phi(a)
    m = mu * Pa + s * pa
    r = C / np.outer(s, s)
    A1, A2 = a[:, None] * np.ones_like(r), a[None, :] * np.ones_like(r)
    rr = np.clip(r, -0.999999, 0.999999); q = np.sqrt(1 - rr * rr)
    # E[u1+ u2+] in standardized form: u_i = s_i (a_i + z_i)
    P2 = Phi2(A1, A2, rr)                        # P(z1 > -a1, z2 > -a2) = Phi2(a1, a2, r)
    t1 = (A1 * A2 + rr) * P2
    t2 = A1 * phi(A2) * ndtr((A1 - rr * A2) / q) + A2 * phi(A1) * ndtr((A2 - rr * A1) / q)
    t3 = q / np.sqrt(2 * np.pi) * phi(np.sqrt(np.maximum((A1**2 - 2*rr*A1*A2 + A2**2) / (q*q), 0)))
    E = np.outer(s, s) * (t1 + t2 + t3)
    d = (mu**2 + s**2) * Pa + mu * s * pa
    np.fill_diagonal(E, d)
    return m, E - np.outer(m, m), Pa

def propagate(mx, Cx, Wi, Wh, Wo):
    """mx (T,), Cx (T,T): Gaussian input. Returns mean, cov of logits."""
    n = T + H
    m = np.zeros(n); C = np.zeros((n, n)); m[:T] = mx; C[:T, :T] = Cx
    for t in range(T):
        B = np.zeros((H, n)); B[:, T:] = Wh; B[:, t] += Wi
        mu = B @ m; Cuv = B @ C; Cuu = Cuv @ B.T
        mh, Chh, Pa = relu_moments(mu, Cuu)
        Chx = Pa[:, None] * Cuv[:, :T]
        m = np.concatenate([m[:T], mh]); C2 = np.zeros((n, n))
        C2[:T, :T] = C[:T, :T]; C2[T:, :T] = Chx; C2[:T, T:] = Chx.T; C2[T:, T:] = Chh
        C = C2
    mh, Chh = m[T:], C[T:, T:]
    return Wo @ mh, Wo @ Chh @ Wo.T

def p_argmax(mL, CL, tgt, method="mvn"):
    D = -np.eye(T); D[:, tgt] += 1; D = np.delete(D, tgt, 0)
    md, Cd = D @ mL, D @ CL @ D.T + 1e-9 * np.eye(T - 1)
    from scipy.stats import multivariate_normal
    return multivariate_normal(mean=np.zeros(T - 1), cov=Cd).cdf(md)   # P(all margins > 0)

if __name__ == "__main__":
    # check relu moments vs MC (diagnostic)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)); C = A @ A.T; mu = rng.standard_normal(4)
    u = rng.multivariate_normal(mu, C, 2_000_000); r = np.maximum(u, 0)
    m, Cr, _ = relu_moments(mu, C)
    print(np.abs(m - r.mean(0)).max(), np.abs(Cr - np.cov(r.T)).max())

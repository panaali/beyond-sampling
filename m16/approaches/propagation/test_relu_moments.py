# diagnosis: check closed-form ReLU moments against Monte Carlo on random Gaussians
import numpy as np
from relu_moments import relu_moments
rng = np.random.default_rng(0)
for trial in range(4):
    n = 4
    L = rng.standard_normal((n, n)); C = L @ L.T / 2 + 0.1 * np.eye(n); mu = rng.standard_normal(n) * 1.5
    if trial == 3: C = np.outer(L[0], L[0]) + 1e-4 * np.eye(n)
    m, cov, _ = relu_moments(mu[None], C[None])
    u = rng.multivariate_normal(mu, C, 4_000_000); r = np.maximum(u, 0)
    print(np.abs(m[0] - r.mean(0)).max(), np.abs(cov[0] - np.cov(r.T)).max())

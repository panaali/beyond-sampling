"""Check cumprop's exact bivariate ReLU moments against MC (and E13's closed form) incl. correlation +-1."""
import numpy as np, sys
from cumprop import relu_layer
r = np.random.default_rng(0)
worst = 0
for trial in range(6):
    n = 6
    if trial < 2:   # rank-1 (correlation +-1), as at step 1 of a scalar-input RNN
        a = r.standard_normal(n); C = np.outer(a, a)
    elif trial < 4:  # rank-2
        B = r.standard_normal((n, 2)); C = B @ B.T
    else:
        B = r.standard_normal((n, n)); C = B @ B.T / n
    mu = r.standard_normal(n) * (0 if trial % 2 == 0 else 1)
    z = r.multivariate_normal(mu, C, size=4_000_000, method="eigh")
    y = np.maximum(z, 0)
    m, cov, _ = relu_layer(mu, C, "gauss")
    em = np.abs(m - y.mean(0)).max(); ec = np.abs(cov - np.cov(y.T)).max()
    worst = max(worst, ec)
    print(f"trial {trial}: max|mean err| {em:.1e}  max|cov err| {ec:.1e} (cov scale {np.abs(cov).max():.2f}; MC se ~{np.abs(cov).max()*2/2000:.0e})")
print("worst cov err", worst)
# deterministic cross-check against E13's closed form (Owen's T), non-degenerate covariances
sys.path.insert(0, "../../m16/approaches/propagation")
from relu_moments import relu_moments
for trial in range(3):
    B = r.standard_normal((8, 8)); C = B @ B.T / 8 + 0.1 * np.eye(8); mu = r.standard_normal(8)
    m1, c1, _ = relu_layer(mu, C, "gauss"); m2, c2, _ = relu_moments(mu[None], C[None])
    print(f"vs E13 closed form: max|mean diff| {np.abs(m1-m2[0]).max():.1e} max|cov diff| {np.abs(c1-c2[0]).max():.1e}")

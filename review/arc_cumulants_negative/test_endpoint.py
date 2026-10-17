"""Sanity check of endpoint_ceiling.endpoint (Gaussian / Edgeworth orthant) on distributions with known answers."""
import sys

import numpy as np
import edgeworth as E
r = np.random.default_rng(0)
# 1) exactly Gaussian margins: all three should agree with the true orthant probability
C = np.array([[1, .5, .2], [.5, 1, .3], [.2, .3, 1.]]); mu = np.array([.3, .5, .1])
z = r.multivariate_normal(mu, C, 2_000_000)
print("gaussian case: true", (z > 0).all(1).mean(), E.endpoint(*E.cumulants(z)))
# 2) sums of k exponentials (skewed, Edgeworth regime): m_i = (Gamma(k) - k*c_i)/sqrt(k), independent
for k in (2, 8, 32):
    g = (r.gamma(k, size=(2_000_000, 2)) - k * np.array([0.8, 0.9])) / np.sqrt(k)
    print(f"gamma k={k}: true {(g > 0).all(1).mean():.4f}", {a: round(b, 4) for a, b in E.endpoint(*E.cumulants(g)).items()})

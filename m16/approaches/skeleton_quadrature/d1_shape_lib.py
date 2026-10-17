"""Shared skeleton-coordinate maps for T103. U in [0,1]^3 (independent uniforms under the top-3 order-statistic law)."""
import numpy as np
from scipy.stats import norm


def vals_from_U(U):
    v1 = norm.ppf(U[:, 0] ** (1 / 10))
    v2 = norm.ppf(U[:, 1] ** (1 / 9) * norm.cdf(v1))
    v3 = norm.ppf(U[:, 2] ** (1 / 8) * norm.cdf(v2))
    return np.stack([v1, v2, v3], 1)


def U_from_vals(v):
    c = norm.cdf(v)
    return np.stack([c[:, 0] ** 10, (c[:, 1] / c[:, 0]) ** 9, (c[:, 2] / c[:, 1]) ** 8], 1)

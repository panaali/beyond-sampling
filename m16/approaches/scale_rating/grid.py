"""T427 pre-registered t grid (model-free): for sequence length T, t_j solves P_T(g23 >= t_j) = P_10(g23 >= s_j) for
T401's grid s_j = .5, 1.0, ..., 3.5 at T = 10, so each grid point is an equally rare condition at every T (at T = 10 it
is T401's grid exactly). P_T(g23 >= t) = int T(T-1) phi(v)(1 - Phi(v)) Phi(v - t)^(T-2) dv (exact quadrature)."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
_G = np.linspace(-7.0, 10.0, 2_000_001)


def p_gap(t, T):
    f = T * (T - 1) * np.exp(-_G ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr(-_G) * ndtr(_G - t) ** (T - 2)
    return float(((f[1:] + f[:-1]) / 2 * np.diff(_G)).sum())


def tgrid(T):
    if T == 10:
        return S
    return tuple(float(brentq(lambda t: np.log(p_gap(t, T)) - np.log(p_gap(s, 10)), 0.01, 8.0, xtol=1e-10)) for s in S)


if __name__ == "__main__":
    for T in (10, 15, 20):
        print(T, [round(t, 4) for t in tgrid(T)], [f"{p_gap(t, T):.3g}" for t in tgrid(T)])

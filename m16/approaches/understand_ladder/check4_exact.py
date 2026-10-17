"""T100 geometry check at T = 4: models with Wh = 0 have class = c+ (x3 > 0) or c- (x3 < 0), so the exact accuracy is
sum_s P(sign x3 = s, 2nd argmax = c_s), computed by 1-D quadrature (scipy.integrate.quad). FX (fibre4) must match it."""
import numpy as np
from scipy import integrate
from scipy.stats import norm

import fibre4
from fiber3 import Ops


def p_second(k, s):
    """P(x_k is the 2nd largest of 4 iid N(0,1) and sign(x_3) = s)."""
    phi, Phi = norm.pdf, norm.cdf
    if k == 3:          # x3 = t is 2nd: exactly one of the other 3 above
        f = lambda t: phi(t) * 3 * (1 - Phi(t)) * Phi(t) ** 2
        return integrate.quad(f, 0, np.inf, epsabs=1e-14)[0] if s > 0 else integrate.quad(f, -np.inf, 0, epsabs=1e-14)[0]
    # x_k = t is 2nd; x3 sign s; the other two (j1, j2) and x3: exactly one of {x3, x_j1, x_j2} above t
    def f(t):
        if s > 0:
            a3 = 1 - Phi(max(t, 0)); b3 = max(Phi(t) - 0.5, 0.0)       # x3 > max(t, 0) ; 0 < x3 < t
        else:
            a3 = max(0.5 - Phi(t), 0.0); b3 = Phi(min(t, 0))           # t < x3 < 0 ; x3 < min(t, 0)
        up, dn = 1 - Phi(t), Phi(t)
        # one above among {x3, j1, j2}: x3 above and j's below, or one j above and x3 + other j below
        return phi(t) * (a3 * dn ** 2 + 2 * up * dn * b3)
    return integrate.quad(f, -np.inf, np.inf, epsabs=1e-14, limit=200)[0]


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    tot = sum(p_second(k, s) for k in range(4) for s in (1, -1))
    print("sanity: sum_k,s P = %.12f (should be 1)" % tot)
    for trial in range(3):
        Wi = rng.standard_normal(4); Wh = np.zeros((4, 4)); Wo = rng.standard_normal((4, 4))
        cp = int(np.argmax(Wo @ np.maximum(Wi, 0))) if (Wi > 0).any() else 0
        cm = int(np.argmax(Wo @ np.maximum(-Wi, 0))) if (Wi < 0).any() else 0
        ex = p_second(cp, 1) + p_second(cm, -1)
        o = Ops(); fx = fibre4.est_fibre4((Wi, Wh, Wo), "FX", o)
        print(f"trial {trial}: exact {ex:.10f} FX {fx:.10f} diff {fx - ex:+.2e}")

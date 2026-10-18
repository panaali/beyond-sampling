"""Check prefix_mass.orth4 (Plackett from the identity, rho = sin(phi) per pair) against Plackett along a DIFFERENT
path: from C0 = C with Y_4 decoupled (exact P3/2) to C, by adaptive quad. Random and thin simplicial cones."""
import warnings

import numpy as np
from scipy.integrate import quad

import prefix_mass as P

warnings.simplefilter("ignore")
rng = np.random.default_rng(3); worst = 0
for tr in range(12):
    R = rng.standard_normal((4, 4))
    if tr >= 6:
        R[:, 1] = R[:, 0] + 10 ** -(tr - 4) * rng.standard_normal(4)       # thin: two rays 1e-2 .. 1e-7 apart
    N, TH = P.normals_angles(R); C = N @ N.T; a = np.pi / 2 - TH
    p = P.orth4(C[None], TH[None])[0][0]
    C0 = C.copy(); C0[3, :3] = 0; C0[:3, 3] = 0

    def dP(s):
        M = (1 - s) * C0 + s * C; tot = 0
        for i in range(3):
            aa, bb = [i, 3], [x for x in range(3) if x != i]
            Sc = M[np.ix_(bb, bb)] - M[np.ix_(aa, bb)].T @ np.linalg.solve(M[np.ix_(aa, aa)], M[np.ix_(aa, bb)])
            r = Sc[0, 1] / np.sqrt(Sc[0, 0] * Sc[1, 1])
            tot += C[i, 3] * (0.25 + np.arcsin(r) / (2 * np.pi)) / (2 * np.pi * np.sqrt(1 - M[i, 3] ** 2))
        return tot
    q = 0.5 * (0.125 + (a[0, 1] + a[0, 2] + a[1, 2]) / (4 * np.pi)) + quad(dP, 0, 1, epsabs=1e-17, epsrel=1e-14,
                                                                            limit=2000)[0]
    worst = max(worst, abs(p - q)); print(tr, f"{p:.15f} {q:.15f} diff {p - q:.1e}")
print("worst abs diff", worst)

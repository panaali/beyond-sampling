"""Cancellation factor of the tie margin vs T (link to INDEX H-E9): at R23 face points (x_a = x_b), the
logit margin m = L_a - L_b = sum_j J_j y_j exactly (gated Jacobian, homogeneity). Factor = sum|J_j y_j|/|m|.
usage: python cancel_T.py"""
import sys
import numpy as np
import core as C
sys.path.insert(0, "/home/user/clean-neurons/m16")
from common import Wi, Wh, Wo


def jac(M, x):
    Wi, Wh, Wo = M; n, T = x.shape; H = len(Wi)
    h = np.zeros((n, H)); J = np.zeros((n, H, T))
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        Jz = np.einsum("ij,njk->nik", Wh, J); Jz[:, :, t] += Wi[None]
        g = z > 0; h = np.where(g, z, 0); J = np.where(g[:, :, None], Jz, 0)
    return np.einsum("oi,nik->nok", Wo, J)


if __name__ == "__main__":
  for name, M in [(f"M16_{T}", C.zoo(16, T)) for T in (3, 4, 5, 6, 8)] + [("orig", (Wi, Wh, Wo))]:
      T = M[2].shape[0]
      U = C.sobol(T, 4096, 9)
      y, a, b, tau = C.face_points(T, "23", U)
      keep = y[np.arange(len(y)), a] > 0
      y, a, b = y[keep], a[keep], b[keep]; r = np.arange(len(y))
      J = jac(M, y + 1e-9 * ((np.arange(T)[None] == a[:, None]) * 1.0))
      G = J[r, a] - J[r, b]                      # (n,T) margin weights
      terms = G * y
      m = terms.sum(1)
      f = np.abs(terms).sum(1) / np.abs(m)
      print(f"{name}: median cancellation factor {np.median(f):.1f}  (IQR {np.percentile(f,25):.1f}-{np.percentile(f,75):.1f}); "
            f"sd(margin)={m.std():.2f}, mean sum|terms|={np.abs(terms).sum(1).mean():.1f}")

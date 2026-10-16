"""Where does the face-integrand variance come from? Decompose Var(F) by configuration and by blind (u<0) faces,
and measure the tail. usage: python diag_var.py T n"""
import sys
import numpy as np
import core as C
T = int(sys.argv[1]); n = int(sys.argv[2])
if T == 10:
    sys.path.insert(0, "/home/user/clean-neurons/m16"); from common import Wi, Wh, Wo; M = (Wi, Wh, Wo)
else:
    M = C.zoo(16, T)
_, Z23, Z12 = C.tables(T)
for kind, Z in (("23", Z23), ("12", Z12)):
    U = C.sobol(T, n, 77)
    y, a, b, tau = C.face_points(T, kind, U)
    F, cost, cells = C.walk_mass(M, y, a, b, tau, kind)
    u = y[np.arange(n), a]
    cfg = np.minimum((U[:, 0] * len(C.configs(T, kind))).astype(int), len(C.configs(T, kind)) - 1)
    tot = F.var()
    means = np.array([F[cfg == c].mean() for c in np.unique(cfg)])
    between = means.var()
    s = np.sort(F)[::-1]
    print(f"T={T} R{kind}: Z*E[F]={1e2*Z*F.mean():.3f}pp  CV={F.std()/F.mean():.2f}  between-config share={between/tot:.2f}  "
          f"u<0 faces: frac {np.mean(u<0):.3f}, share of mean {F[u<0].sum()/F.sum():.2f}, share of 2nd moment {np.sum(F[u<0]**2)/np.sum(F**2):.2f}; "
          f"top1% faces carry {s[:n//100].sum()/s.sum():.2f} of mean, {np.sum(s[:n//100]**2)/np.sum(s**2):.2f} of 2nd moment; cells {cells.mean():.1f}")

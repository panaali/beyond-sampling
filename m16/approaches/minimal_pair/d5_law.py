"""D5: decompose the iid tie-face gain as  gain = (1-p) / (err * CV^2 * c)  along T (H=16, seeds 0-4).
CV = Neyman-combined sd per sqrt(cost) / (err * sqrt(c)), c = mean honest cost per face (walk.py)."""
import json, sys
import numpy as np
import core as C, walk as W
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
res = {}
for T in (3, 4, 5, 6, 8, 10):
  for s in range(5):
    m = C.zoo(16, T, s); _, Z23, Z12 = C.tables(T); err = 0; S = 0; cs = []; f2neg = 0; f2 = 0
    for kind, Z in (("23", Z23), ("12", Z12)):
        U = np.random.default_rng(55 + 10 * T + s + (kind == "12")).random((N, T))
        y, a, b, tau = C.face_points(T, kind, U)
        F, c, _ = W.walk(m, y, a, b, tau, kind)
        err += Z * F.mean(); S += Z * F.std() * np.sqrt(c.mean()); cs.append((Z, c.mean()))
        neg = y[np.arange(N), a] < 0
        f2neg += Z ** 2 * (F[neg] ** 2).sum() / N; f2 += Z ** 2 * (F ** 2).sum() / N
    p = 1 - err; gain = p * (1 - p) / S ** 2
    cbar = (sum(Z * np.sqrt(c) for Z, c in cs) / (Z23 + Z12)) ** 2
    CV = S / (err * np.sqrt(cbar))
    res[f"{T}_{s}"] = dict(err=err, gain=gain, CV=CV, c=cbar, negshare=f2neg / f2)
    print(f"T{T} s{s}: err {err:.5f}  iid gain {gain:7.1f}  CV {CV:.2f}  c {cbar:5.2f}  (1-p)/(err CV^2 c) {(1-p)/(err*CV**2*cbar):7.1f}"
          f"  E[F^2] share u<0 {f2neg/f2:.2f}", flush=True)
json.dump(res, open("d5_law.json", "w"), indent=1)

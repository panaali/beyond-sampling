"""Check walk.walk == core.walk_mass (mass) and compare charged cost; also the hom (Student-t) variant."""
import time
import numpy as np
import core as C, walk as W, hom
for T, s in ((4, 0), (5, 2), (10, 0)):
    m = C.zoo(16, T, s)
    for kind in ("23", "12"):
        U = np.random.default_rng(1).random((4096, T))
        y, a, b, tau = C.face_points(T, kind, U)
        t0 = time.process_time(); F0, c0, n0 = C.walk_mass(m, y, a, b, tau, kind); t1 = time.process_time()
        F1, c1, n1 = W.walk(m, y, a, b, tau, kind); t2 = time.process_time()
        w, a2, b2, tw = hom.faces_hom(T, kind, U)
        G0, _, _ = hom.walk_mass_hom(m, w, a2, b2, tw, kind)
        G1, _, _ = W.walk(m, w, a2, b2, tw, kind, cdf=hom.tcdf(T))
        print(T, kind, "max|dF|", np.abs(F0 - F1).max(), "max|dG|", np.abs(G0 - G1).max(), "cells eq", (n0 == n1).mean(),
              f"cost core {c0.mean():.2f} walk {c1.mean():.2f} full {2*n0.mean():.2f} | cpu {t1-t0:.2f}s vs {t2-t1:.2f}s")

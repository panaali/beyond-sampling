"""Check E[F_hom] = E[F] and compare per-face variance (iid faces), T=4,5 seeds 0-4."""
import numpy as np
import core as C, hom
N = 32768
for T in (4, 5):
  for s in range(5):
    m = C.zoo(16, T, s); _, Z23, Z12 = C.tables(T); out = []
    e0 = e1 = 0; v0 = v1 = 0; c0 = c1 = 0
    for kind, Z in (("23", Z23), ("12", Z12)):
        U = np.random.default_rng(31 + s + 10 * T + (kind == "12")).random((N, T))
        y, a, b, tau = C.face_points(T, kind, U)
        F, c, _ = C.walk_mass(m, y, a, b, tau, kind)
        w, a2, b2, tw = hom.faces_hom(T, kind, U)
        G, c2, _ = hom.walk_mass_hom(m, w, a2, b2, tw, kind)
        e0 += Z * F.mean(); e1 += Z * G.mean()
        v0 += Z * F.std() * np.sqrt(c.mean()); v1 += Z * G.std() * np.sqrt(c2.mean())
        out.append(f"{kind}: sdF {F.std():.2e} sdG {G.std():.2e} cost {c.mean():.2f}/{c2.mean():.2f}")
    p = 1 - e0
    print(f"T{T} s{s} err F {e0:.5f} G {e1:.5f} | iid Neyman gain F {p*(1-p)/v0**2:.1f}x  G {p*(1-p)/v1**2:.1f}x |", *out, flush=True)

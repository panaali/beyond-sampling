"""T195 phase 2 controls for prune.cap_bound (outward mass bound): narrow simplicial cones at scales 1e-1..1e-15
(d = 3: Van Oosterom-Strackee solid angle; d = 4, 5: estT.orthant incl. its thin rule), exactly degenerate cones (repeated / collinear rays),
near-half-space cones, and 519 recipe sub-cones vs estT.mass. Prints violations (bound < reference)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro")]
sys.argv = sys.argv[:1]
import numpy as np, estT, prune
from m43 import Ops
rng = np.random.default_rng(0); viol = 0; n = 0; rows = []
for d in (3, 4, 5):
    for e in 10.0 ** -np.arange(1, 16):
        for _ in range(5):
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            R = u + e * rng.standard_normal((d, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            c = estT.Cone(R, None, None); b = prune.cap_bound(c, Ops())
            try:
                if d == 3:          # Van Oosterom-Strackee solid angle (no 1/8 + sum asin cancellation for tiny cones)
                    a, b_, c_ = R; m = abs(2 * np.arctan2(abs(np.linalg.det(R)), 1 + a @ b_ + b_ @ c_ + c_ @ a)) / (4 * np.pi)
                else:
                    m = estT.orthant(R, Ops()) if abs(np.linalg.det(R)) > 0 else 0.0
            except Exception:
                m = float("nan")
            n += 1; bad = not (b >= m) and not np.isnan(m); viol += bad
            if bad or _ == 0:
                rows.append((d, e, m, b))
        R = np.tile(u, (d, 1)); b = prune.cap_bound(estT.Cone(R, None, None), Ops()); rows.append((d, "repeated rays", 0.0, b)); n += 1; viol += not (b > 0)
        R = np.vstack([u, u, -u][:d] + [u] * (d - 3)); b = prune.cap_bound(estT.Cone(R, None, None), Ops()); rows.append((d, "antipodal rays", "-", b)); n += 1; viol += not (b == 1.0)
for r in rows:
    print(r)
cnt = 0
for T in (3, 4):
    for h in estT.halves(T)[:40]:
        c = h["cone"]; rg = np.random.default_rng(cnt)
        for _ in range(3):
            ci, co = estT.clip(c, rg.standard_normal(T), Ops())
            for q in (ci, co, c):
                if q is not None:
                    m = estT.mass(q, Ops()); b = prune.cap_bound(q, Ops()); n += 1; cnt += 1; viol += b < m
print("checks", n, "violations", viol)

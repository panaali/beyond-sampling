"""T185 validation of the 5-D orthant: (a) random simplicial cones vs 1e7 MC; (b) thin rule vs Plackett at
1e-6 <= det < 1e-3 and rule convergence; (c) a 5-D chamber half (non-simplicial) vs 1e7 MC. -> val5.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro")]
import numpy as np, estT
from m43 import Ops
rng = np.random.default_rng(11); out = {}
def mc(N, n=10_000_000):
    h = 0
    for _ in range(n // 500_000):
        x = rng.standard_normal((500_000, 5)); h += np.all(x @ N.T >= 0, 1).sum()
    p = h / n; return p, (p * (1 - p) / n) ** .5
zs = []
for k in range(6):
    c = rng.standard_normal(5) * (k % 3); R = c + rng.standard_normal((5, 5)); R /= np.linalg.norm(R, axis=1)[:, None]
    if np.linalg.det(R) < 0: R = R[[1, 0, 2, 3, 4]]
    v = estT.orthant(R, Ops()); p, se = mc(np.linalg.inv(R.T)); zs.append(((v - p) / se, v))
out["simplex_vs_mc_z"] = zs
H = [h for h in estT.halves(5)][:3]
out["half_vs_mc_z"] = [((estT.mass(h["cone"], Ops()) - mc(h["cone"].N)[0]) / mc(h["cone"].N)[1]) for h in H]
out["halves_sum_minus_1"] = sum(estT.mass(h["cone"], Ops()) for h in estT.halves(5)) - 1
# thin rule vs Plackett, and rule convergence
rows = []
for sc in (3e-2, 1e-2, 3e-3):
    for _ in range(20):
        c = rng.standard_normal(5); R = c + sc * rng.standard_normal((5, 5)); R /= np.linalg.norm(R, axis=1)[:, None]
        if np.linalg.det(R) < 0: R = R[[1, 0, 2, 3, 4]]
        estT.THIN_DET5[0] = 0; estT.set_rules5(6, 256); pl = estT.orthant(R, Ops()); estT.set_rules5(6, 128); pl128 = estT.orthant(R, Ops())
        estT.set_rules5(6, 64); pl64 = estT.orthant(R, Ops())
        estT.THIN_DET5[0] = 1; th = {}
        for n in (4, 6, 10):
            estT.set_rules5(n, 128); th[n] = estT.orthant(R, Ops())
        rows.append((abs(np.linalg.det(R)), pl, pl128 - pl, pl64 - pl, th[4] - th[10], th[6] - th[10], th[10] - pl))
estT.THIN_DET5[0] = 1e-6; estT.set_rules5()
A = np.array(rows); out["thin_rows(det,pl256,d128,d64,th4-10,th6-10,th10-pl)"] = A.tolist()
rel = lambda c: float(np.max(np.abs(A[:, c]) / np.abs(A[:, 1])))
out["max_rel"] = {"log128": rel(2), "log64": rel(3), "thin4": rel(4), "thin6": rel(5), "thin10_vs_plackett": rel(6)}
print("simplex z", [round(z, 2) for z, _ in zs]); print("half z", np.round(out["half_vs_mc_z"], 2), "halves sum-1 %.2e" % out["halves_sum_minus_1"])
print("det range %.1e..%.1e" % (A[:, 0].min(), A[:, 0].max()), out["max_rel"])
json.dump(out, open(os.path.join(HERE, "val5.json"), "w"), indent=1)

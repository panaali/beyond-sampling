"""T353 POST HOC (DRI request, not pre-registered): tilt-only arm with walk-state reuse (cont_h2.est(spread=0)) on DEV2 and
DEV3. The decision rule is unchanged; this is reported only."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cont_h2  # noqa: E402
D = os.path.join(HERE, "../../../../data/oracle_floor"); out = {}
for panel in ("dev2", "dev3"):
    z = np.load(os.path.join(D, f"{panel}_weights.npz")); tr = json.load(open(os.path.join(D, f"{panel}_truths.json")))["truths"]
    e, Cs, p = [], [], []
    for k, nm in enumerate(z["names"]):
        m = tuple(z[a][k].astype(np.float64) for a in ("Wi", "Wh", "Wo")); a, f = cont_h2.est(m, spread=0.0)
        e.append(100 * (a - tr[str(nm)]["acc"])); Cs.append(f / 5948); p.append(tr[str(nm)]["acc"])
    e, p = np.array(e), np.array(p); mse = float((e ** 2).mean()); C = float(np.mean(Cs)); pq = float((p * (1 - p)).mean())
    out[panel] = dict(n=len(e), bias=float(e.mean()), sd=float(e.std(ddof=1)), rms=float(np.sqrt(mse)), C=C, mse_x_C=mse * C, F=mse / 1e4 * max(C, 1000) / pq)
    print(panel, out[panel], flush=True)
json.dump(out, open(os.path.join(HERE, "posthoc_a2cheap.json"), "w"), indent=1)

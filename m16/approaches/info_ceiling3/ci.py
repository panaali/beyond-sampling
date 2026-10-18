"""T332: bootstrap CI (models resampled) of the held-out RMS for the primary ridge and the offset-only fit, and the
implied F at 2.2k / 5.4k FPE. Reported, not deciding."""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = np.load("results/features.npz"); R = json.load(open("results/learn.json"))
X, est, truth, names = D["X"], D["est"], D["truth"], D["names"]
y = (truth - est) * 100; tr = np.array([n.startswith("fused_") for n in names]); te = ~tr
m = make_pipeline(StandardScaler(), Ridge(alpha=R["ridge (primary)"]["best"]["ridge__alpha"])).fit(X[tr], y[tr])
errs = {"ridge": (m.predict(X) - y)[te], "offset": (y[tr].mean() - y)[te], "raw": -y[te]}
pq = (truth[te] * (1 - truth[te])).mean()
rng = np.random.default_rng(332); out = {"mean_pq_test": float(pq)}
for k, e in errs.items():
    b = [np.sqrt(np.mean(e[i] ** 2)) for i in (rng.integers(0, len(e), len(e)) for _ in range(10000))]
    r = float(np.sqrt(np.mean(e ** 2)))
    out[k] = {"rms": r, "ci95": [float(v) for v in np.percentile(b, [2.5, 97.5])],
              "F_2.2k": (r / 100) ** 2 * 2200 / pq, "F_5.4k": (r / 100) ** 2 * 5431 / pq}
json.dump(out, open("results/ci.json", "w"), indent=1); print(json.dumps(out, indent=1))

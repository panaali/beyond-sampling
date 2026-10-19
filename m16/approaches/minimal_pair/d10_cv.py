"""T389 development: held-out value of a (stratum, a, b) allocation table fitted on half of 40 DEV2 models.
Per stratum, var*cost: proportional (frozen recipe) = sum w(s^2 + (m - mbar)^2) * sum w c;
table q ~ w s_hat / sqrt(c_hat): (sum w^2 s^2 / q)(sum q c); oracle q ~ w s / sqrt(c).
Strata combined by oracle Neyman in every arm (the frozen recipe's pilot does this)."""
import numpy as np
X = np.concatenate([np.load("../../../data/minimal_pair_d10_0_20.npy"), np.load("../../../data/minimal_pair_d10_20_40.npy")])
from estimator import strata, stratum_mass
M = np.array([stratum_mass(10, k, h) for k, h in strata(10)])
def vc(w, mu, s, c, q):
    ok = w > 0
    w, mu, s, c, q = w[ok], mu[ok], s[ok], c[ok], q[ok]
    if q is None: return None
    return (np.sum(w ** 2 * s ** 2 / q)) * np.sum(q * c)
def arms(x, table):
    tot = {"prop": 0, "table": 0, "oracle": 0}
    for j in range(5):
        w, mu, s, c = x[j, :, 0], x[j, :, 1], x[j, :, 2], x[j, :, 3]
        ok = w > 0; mbar = np.sum(w * mu)
        vp = np.sum(w[ok] * (s[ok] ** 2 + (mu[ok] - mbar) ** 2)) * np.sum(w[ok] * c[ok])
        qt = np.where(ok, w * np.maximum(table[j], 1e-12) / np.sqrt(np.maximum(c, 1e-9)), 0); qt /= qt.sum()
        qo = np.where(ok, w * np.maximum(s, 1e-12) / np.sqrt(np.maximum(c, 1e-9)), 0); qo /= qo.sum()
        for k, v in (("prop", vp), ("table", vc(w, mu, s, c, qt)), ("oracle", vc(w, mu, s, c, qo))):
            tot[k] += M[j] * np.sqrt(v)
    return {k: v ** 2 for k, v in tot.items()}
def fit(models):
    # relative sd per cell within stratum, normalised per model by the stratum's mean sd, averaged (geometric) over models
    t = np.zeros((5, 100))
    for j in range(5):
        s = models[:, j, :, 2]; w = models[:, j, :, 0]
        rel = s / np.maximum((w * s).sum(1, keepdims=True), 1e-30)
        t[j] = np.exp(np.mean(np.log(np.maximum(rel, 1e-6)), 0))
    return t
for name, tr, te in (("fit A test B", X[:20], X[20:]), ("fit B test A", X[20:], X[:20])):
    tab = fit(tr); r = np.array([[a["prop"] / a["table"], a["prop"] / a["oracle"]] for a in (arms(x, tab) for x in te)])
    print(f"{name}: gain over frozen allocation  table median {np.median(r[:,0]):.2f} [min {r[:,0].min():.2f}, max {r[:,0].max():.2f}]"
          f"   oracle median {np.median(r[:,1]):.2f} [min {r[:,1].min():.2f}, max {r[:,1].max():.2f}]")

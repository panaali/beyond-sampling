"""T334: check, bias/MSE, deltas and bootstrap CIs (models resampled; per-model MC noise drawn from paired covariance)."""
import json
import numpy as np

D = json.load(open("results/pointwise_all.json")); names = sorted(D)
M = np.array([D[n]["mean"] for n in names]); C = np.array([D[n]["cov"] for n in names])
e = np.array([D[n]["est_e"] for n in names]); tr = np.array([D[n]["truth"] for n in names])
se = np.sqrt(C[:, 0, 0]); z = (M[:, 0] - e) / se
chk = {"max_abs_z": float(np.abs(z).max()), "rms_diff_pp": float(np.sqrt(np.mean((M[:, 0] - e) ** 2)) * 100),
       "mean_se_pp": float(se.mean() * 100)}
chk["pass"] = bool(chk["max_abs_z"] <= 4 and chk["rms_diff_pp"] <= 2 * chk["mean_se_pp"])


def stats(Mx):
    err = (Mx - tr[:, None]) * 100
    b, mse = err.mean(0), (err ** 2).mean(0)
    return b, mse, b[1:] - b[0], (mse[1:] - mse[0]) / mse[0]


b, mse, db, dm = stats(M)
rng = np.random.default_rng(334); L = np.linalg.cholesky(C + 1e-18 * np.eye(3)); BD, BM = [], []
for _ in range(10000):
    i = rng.integers(0, len(names), len(names))
    Mx = M[i] + np.einsum("mij,mj->mi", L[i], rng.standard_normal((len(i), 3)))
    tr_i = tr[i]; err = (Mx - tr_i[:, None]) * 100
    bb, mm = err.mean(0), (err ** 2).mean(0)
    BD.append(bb[1:] - bb[0]); BM.append((mm[1:] - mm[0]) / mm[0])
ci = lambda A: np.percentile(np.array(A), [2.5, 97.5], 0).T.tolist()
cd, cm = ci(BD), ci(BM)
t324 = {"product": (-.57, -.40), "tmax": (-.18, -.18)}
res = {"n_models": len(names), "check": chk,
       "bias_pp": dict(zip(["V1", "product_off", "tmax_off"], b.tolist())),
       "rms_pp": dict(zip(["V1", "product_off", "tmax_off"], np.sqrt(mse).tolist()))}
for j, k in enumerate(["product", "tmax"]):
    res[k] = {"d_bias_pp": float(db[j]), "d_bias_ci": cd[j], "d_mse_frac": float(dm[j]), "d_mse_ci": cm[j],
              "t324": t324[k],
              "verdict": "CONFIRMED" if (cd[j][0] <= t324[k][0] <= cd[j][1] and cm[j][0] <= t324[k][1] <= cm[j][1])
              else "DISCREPANCY"}
json.dump(res, open("results/summary.json", "w"), indent=1); print(json.dumps(res, indent=1))

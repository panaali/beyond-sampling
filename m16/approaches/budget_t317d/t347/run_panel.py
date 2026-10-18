"""T347 out-of-sample test on any panel npz (Wi, Wh, Wo, names): D_pred (frozen tilt.py) vs D_m = V1 - J1 measured by
budget.py's pointwise form (paired 2e5 exact-law inputs per model, seed 347000 + index; no truths needed).
usage: python run_panel.py WEIGHTS.npz OUT.json [expected_sha256_of_arrays]"""
import hashlib, json, os, sys
import numpy as np
path, outp = sys.argv[1], sys.argv[2]; sha = sys.argv[3] if len(sys.argv) > 3 else None
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
sys.argv = sys.argv[:1]
import tilt  # noqa: E402
import budget as B  # noqa: E402
z = np.load(path)
h = hashlib.sha256(z["Wi"].tobytes() + z["Wh"].tobytes() + z["Wo"].tobytes()).hexdigest(); print("weights sha256", h)
if sha:
    assert h == sha, "hash mismatch"
out = {"weights_sha256": h, "per_model": {}}
for k, nm in enumerate(z["names"]):
    m = tuple(z[a][k].astype(np.float64) for a in ("Wi", "Wh", "Wo"))
    rng = np.random.default_rng(347000 + k); P = B.Point(rng.standard_normal((B.N, 10)))
    Dm = 100 * (B.variant(m, P) - B.variant(m, P, product_on=False)).mean(); d, info = tilt.D_pred(m)
    out["per_model"][str(nm)] = dict(D_pred=d, D_meas=float(Dm)); print(nm, "pred %+.3f meas %+.3f" % (d, Dm), flush=True)
P_ = np.array([v["D_pred"] for v in out["per_model"].values()]); M = np.array([v["D_meas"] for v in out["per_model"].values()])
out["summary"] = dict(n=len(M), pred_mean=float(P_.mean()), meas_mean=float(M.mean()), panel_err=float(P_.mean() - M.mean()),
                      primary_go=bool(abs(P_.mean() - M.mean()) <= .1), mean_abs_err=float(np.abs(P_ - M).mean()), corr=float(np.corrcoef(P_, M)[0, 1]),
                      meas_sd=float(M.std(ddof=1)), meas_mean_se=float(M.std(ddof=1) / np.sqrt(len(M))))
print(out["summary"]); json.dump(out, open(outp, "w"), indent=1)

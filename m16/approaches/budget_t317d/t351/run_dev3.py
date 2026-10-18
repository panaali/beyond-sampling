"""T353 pre-registered DEV3 check (frozen before DEV3 exists): estimator_h2 (T353) vs cont_f (T317f) on the same models,
no retuning. usage: python run_dev3.py WEIGHTS.npz TRUTHS.json OUT.json [weights_sha256]
WEIGHTS.npz: Wi (n,16), Wh (n,16,16), Wo (n,10,16), names. TRUTHS.json: {"truths": {name: {"acc": ...}}} (T192 format)."""
import hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cont_f, estimator_h2  # noqa: E402
wp, tp, op = sys.argv[1:4]; sha = sys.argv[4] if len(sys.argv) > 4 else None
z = np.load(wp); h = hashlib.sha256(z["Wi"].tobytes() + z["Wh"].tobytes() + z["Wo"].tobytes()).hexdigest(); print("weights sha256", h)
if sha:
    assert h == sha
tr = json.load(open(tp))["truths"]; rows = {}
for k, nm in enumerate(z["names"]):
    m = tuple(z[a][k].astype(np.float64) for a in ("Wi", "Wh", "Wo")); nm = str(nm)
    a, fa = cont_f.est(m); b, fb = estimator_h2.est(m)
    rows[nm] = dict(truth=tr[nm]["acc"], T317f=a, C_T317f=fa / 5948, T353=b, C_T353=fb / 5948)
    print(nm, "T317f %+.3f T353 %+.3f" % (100 * (a - tr[nm]["acc"]), 100 * (b - tr[nm]["acc"])), flush=True)
N = list(rows); p = np.array([rows[n]["truth"] for n in N]); pq = float((p * (1 - p)).mean()); S = {"n": len(N), "weights_sha256": h}
for key in ("T317f", "T353"):
    e = 100 * (np.array([rows[n][key] for n in N]) - p); C = float(np.mean([rows[n]["C_" + key] for n in N])); mse = float((e ** 2).mean())
    S[key] = dict(bias=float(e.mean()), sd=float(e.std(ddof=1)), rms=float(np.sqrt(mse)), C=C, mse_x_C=mse * C, F=float(mse / 1e4 * max(C, 1000) / pq))
S["confirms"] = bool(S["T353"]["mse_x_C"] < S["T317f"]["mse_x_C"])
json.dump(dict(per_model=rows, summary=S), open(op, "w"), indent=1); print(S)

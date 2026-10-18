"""T332: pre-registered learners on features.npz (train fused_s0..39; test torch_s0..39 + LB3 seeds). Diagnostic."""
import json
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = np.load("results/features.npz")
X, est, truth, names, fn = D["X"], D["est"], D["truth"], D["names"], list(D["fnames"])
y = (truth - est) * 100                                         # residual, pp
tr = np.array([n.startswith("fused_") for n in names])
te = ~tr
torch = np.array([n.startswith("torch_") for n in names]); seeds = np.array([n.startswith("seed") for n in names])
rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
inner = KFold(5, shuffle=True, random_state=0)


def report(pred):
    err = pred - y                                             # error of the corrected estimate, pp
    return {"held_out_rms": rms(err[te]), "torch_rms": rms(err[torch]), "seeds_rms": rms(err[seeds]),
            "train_rms": rms(err[tr]), "held_out_mean_err": float(err[te].mean())}


res = {"raw_T317e": report(np.zeros_like(y)), "offset_only": report(np.full_like(y, y[tr].mean()))}
res["offset_only"]["inner_cv_rms"] = rms(np.concatenate(
    [y[tr][v] - y[tr][t].mean() for t, v in inner.split(X[tr])]))
learners = {
    "ridge (primary)": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-3, 9, 13)}),
    "gbt (secondary)": (GradientBoostingRegressor(max_depth=2, learning_rate=.05, subsample=.8, random_state=0),
                        {"n_estimators": [50, 100, 200, 400]}),
}
for name, (m, grid) in learners.items():
    g = GridSearchCV(m, grid, cv=inner, scoring="neg_root_mean_squared_error").fit(X[tr], y[tr])
    r = report(g.predict(X))
    r.update(inner_cv_rms=float(-g.best_score_), best={k: float(v) for k, v in g.best_params_.items()})
    if name.startswith("ridge"):
        rg = g.best_estimator_.named_steps["ridge"]
        r["coef_std"] = dict(sorted(zip(fn, map(float, rg.coef_)), key=lambda kv: -abs(kv[1])))
        r["intercept_pp"] = float(rg.intercept_)
    res[name] = r
h = res["ridge (primary)"]["held_out_rms"]
res["verdict"] = "GO" if h <= .50 else ("STOP" if h >= .62 else "neither")
json.dump(res, open("results/learn.json", "w"), indent=1)
for k, v in res.items():
    print(k, v if isinstance(v, str) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                           if kk != "coef_std"})
print("top ridge coefs:", list(res["ridge (primary)"]["coef_std"].items())[:8])

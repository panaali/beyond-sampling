"""T314 learners with nested CV (inner 5-fold on train only). Main family A (<= 2.5k FPE); secondary family B
adds probe-v1 (~6.3k FPE total)."""
import json
import warnings
import numpy as np
from scipy.special import ndtr
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings("ignore")
d = np.load("results/perclass.npz", allow_pickle=True)
names = [str(n) for n in d["names"]]
P, S = d["P"], d["S"]
y = 100 * S[:, 1]
tb = 100 * S[:, 0]
pr = 100 * S[:, 2]
train = np.array([n.startswith("fused_") for n in names])
torch = np.array([n.startswith("torch_") for n in names])
seeds = np.array([n.startswith("seed") for n in names])
test = torch | seeds
m_tie, dm_tie, tau, M, m_med = (P[:, i] for i in range(5))
SC = (0.25, 0.5, 1, 2, 4, 8)


def agg_feats():
    f = [ndtr(-m_med / s).mean(1) for s in SC] + [ndtr(-m_tie / s).mean(1) for s in SC]
    tc = np.minimum(tau, 3)
    f += [tc.mean(1), (tau == 0).mean(1)]
    f += [np.quantile(m_med, q, 1) for q in (.1, .5, .9)] + [np.quantile(dm_tie, q, 1) for q in (.1, .5, .9)]
    f += [M.mean(1), tb]
    return np.stack(f, 1)


def basis_feats():
    tc = np.minimum(tau, 3)
    b = [ndtr(-m_med / s) for s in SC] + [ndtr(-m_tie / s) for s in SC]
    b += [(tau == 0).astype(float), tc, M, ndtr(-m_med) * tc, ndtr(-m_tie) * M]
    return np.stack([x.mean(1) for x in b], 1)


def full_vec():
    return np.concatenate([P.reshape(len(P), -1), tb[:, None]], 1)


inner = KFold(5, shuffle=True, random_state=0)
LEARNERS = {
    "ridge_full": (full_vec, make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-3, 9, 13)}),
    "lasso_full": (full_vec, make_pipeline(StandardScaler(), Lasso(max_iter=20000)), {"lasso__alpha": np.logspace(-4, 1, 20)}),
    "gbt_agg": (agg_feats, GradientBoostingRegressor(max_depth=2, learning_rate=.05, subsample=.8, random_state=0),
                {"n_estimators": [50, 100, 200, 400]}),
    "perclass_basis_ridge": (basis_feats, make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-3, 9, 13)}),
}
out = {}


def run(tag, extra=None):
    res = {}
    for name, (fx, est, grid) in LEARNERS.items():
        X = fx()
        if extra is not None:
            X = np.c_[X, extra]
        g = GridSearchCV(est, grid, cv=inner, scoring="neg_root_mean_squared_error").fit(X[train], y[train])
        p = g.predict(X)
        rm = lambda r: float(np.sqrt(np.mean((p[r] - y[r]) ** 2)))
        res[name] = {"inner_cv_rms": float(-g.best_score_), "best": {k: float(v) for k, v in g.best_params_.items()},
                     "test_rms": rm(test), "torch_rms": rm(torch), "seeds_rms": rm(seeds), "train_rms": rm(train),
                     "test_mean_err": float(np.mean(p[test] - y[test]))}
        print(tag, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
    best = min(res, key=lambda k: res[k]["inner_cv_rms"])
    print(tag, "SELECTED by inner CV:", best, "test RMS %.3f" % res[best]["test_rms"])
    return {"learners": res, "selected": best}


out["A_cheap_2.4k"] = run("A")
out["B_plus_probe1_6.3k"] = run("B", extra=pr[:, None])
# reference: T300 estimate alone (raw) and the linear 1-feature fit
e = tb[test] - y[test]
out["tb_raw_test_rms"] = float(np.sqrt(np.mean(e ** 2)))
json.dump(out, open("results/learn.json", "w"), indent=1)
print("T300 raw estimate, test RMS %.3f" % out["tb_raw_test_rms"])

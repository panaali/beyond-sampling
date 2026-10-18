"""T308 pre-registered fit: <= 4-feature OLS, subset by train-only LOO CV, test on torch_s* + LB3 seeds."""
import itertools
import json
import numpy as np

D = json.load(open("results/features.json"))["models"]
names = list(D)
FN = list(D[names[0]]["features"])
X = np.array([[D[n]["features"][f] for f in FN] for n in names])
y = np.array([100 * D[n]["truth"] for n in names])
train = np.array([n.startswith("fused_") for n in names])
torch = np.array([n.startswith("torch_") for n in names])
seeds = np.array([D[n]["panel"] == "lb3" for n in names])
test = torch | seeds
Xs = X.copy()
Xs[:, [0, 8, 9]] *= 100          # estimators in pp


def ols(cols, rows):
    A = np.c_[np.ones(rows.sum()), Xs[rows][:, cols]]
    return np.linalg.lstsq(A, y[rows], rcond=None)[0]


def pred(cols, beta, rows):
    return np.c_[np.ones(rows.sum()), Xs[rows][:, cols]] @ beta


def loo(cols):
    idx = np.where(train)[0]; e = []
    for i in idx:
        r = train.copy(); r[i] = False
        b = ols(cols, r)
        e.append(pred(cols, b, np.eye(len(y), dtype=bool)[i])[0] - y[i])
    return float(np.sqrt(np.mean(np.square(e))))


def report(cols, label):
    b = ols(cols, train)
    rm = lambda rows: float(np.sqrt(np.mean((pred(cols, b, rows) - y[rows]) ** 2)))
    r = {"features": [FN[c] for c in cols], "beta": b.tolist(), "train_loo_rms": loo(cols), "train_fit_rms": rm(train),
         "test_rms": rm(test), "torch_rms": rm(torch), "seeds_rms": rm(seeds),
         "test_mean_err": float(np.mean(pred(cols, b, test) - y[test]))}
    print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "beta"})
    return r


subs = [c for k in range(1, 5) for c in itertools.combinations(range(len(FN)), k)]
scores = sorted((loo(list(c)), c) for c in subs)
out = {"selected": report(list(scores[0][1]), "SELECTED (train LOO)"),
       "two_estimators": report([8, 9], "F9+F10 (pre-specified)"),
       "top5_by_loo": [{"loo": s, "features": [FN[i] for i in c]} for s, c in scores[:5]]}
for f in range(len(FN)):
    out[f"single_{FN[f]}"] = report([f], f"single {FN[f]}")
# raw estimators as-is (no fit)
for f, lab in ((0, "F1_tb"), (8, "F9_rx"), (9, "F10_probe1")):
    e = Xs[test, f] - y[test]
    print(f"raw {lab} on test: RMS {np.sqrt(np.mean(e**2)):.3f} mean {e.mean():+.3f}")
ea, eb = Xs[:, 8] - y, Xs[:, 9] - y
out["rho_rx_probe_errors_all"] = float(np.corrcoef(ea, eb)[0, 1])
print("corr of rx and probe-v1 errors (all 85):", round(out["rho_rx_probe_errors_all"], 3))
json.dump(out, open("results/fit.json", "w"), indent=1)

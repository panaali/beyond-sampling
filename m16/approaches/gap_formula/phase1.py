"""T269 Phase 1: fit P(err | V) by logistic formulas, integrate over the exact order-statistic law.

Non-blind part (M2 > 0) only; blind mass 11/1024 times a measured blind error rate (separately for K=0 and K=1).
Coordinates for quadrature: (M2, z23 = log g23, z12 = log g12); density 720 phi(M1) phi(M2) phi(M3) Phi(M3)^7.
Usage: python phase1.py   (writes results/phase1.json)
"""
import json
import numpy as np
from scipy.special import expit, ndtr, log_ndtr
from scipy.stats import binom
from gfcommon import MODELS, make, ANSWER_KEY_TARGET

LOG720 = np.log(720.0)
PC = np.array([0.1, 0.1, 0.8])          # exact law of the x9 case
BLIND = {0: 1 / 1024, 1: 10 / 1024}       # exact masses of K = 0, 1 (blind = M2 <= 0)


# ------------------------------------------------------------------ features
def feats(fam, M1, M2, M3, K):
    g12, g23 = M1 - M2, M2 - M3
    if fam == "V1":
        a, b = np.log(g23), np.log(g12)
        cols = [np.ones_like(a), a, b, a * a, b * b, a * b]
    else:   # scale-free V2 (and V3 adds K terms)
        a, b, q = np.log(g23 / M2), np.log(g12 / M1), np.log(M2 / M1)
        cols = [np.ones_like(a), a, b, q, a * a, b * b, q * q, a * b, a * q, b * q]
        if fam == "V3":
            k = np.asarray(K, float)
            cols += [(k == 2) * 1.0, (k == 3) * 1.0, np.minimum(k, 8) / 8.0]
    return np.stack(cols, -1)


FAMS = {"V1": 6, "V2": 10, "V3": 13}


def design(fam, V):
    """Per-case block design: features times one-hot(case)."""
    F = feats(fam, V["M1"], V["M2"], V["M3"], V["K"])
    X = np.zeros((len(F), 3 * F.shape[1]))
    for c in range(3):
        m = V["c"] == c
        X[m, c * F.shape[1]:(c + 1) * F.shape[1]] = F[m]
    return X


def logistic(X, y, offset=None, lam=1e-6, iters=30):
    off = np.zeros(len(y)) if offset is None else offset
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        g = lam * w
        Hm = lam * np.eye(len(w))
        for i in range(0, len(y), 250_000):          # chunked to bound memory
            Xi = X[i:i + 250_000]
            p = expit(Xi @ w + off[i:i + 250_000])
            g = g + Xi.T @ (p - y[i:i + 250_000])
            Hm = Hm + (Xi * (p * (1 - p))[:, None]).T @ Xi
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


# ------------------------------------------------------------------ quadrature
def gl(a, b, n):
    u, w = np.polynomial.legendre.leggauss(n)
    return (a + b) / 2 + (b - a) / 2 * u, (b - a) / 2 * w


def grid(n_m=96, n_z=96):
    mA, wA = gl(0.0, 1.5, n_m // 2)
    mB, wB = gl(1.5, 6.0, n_m // 2)
    m2, wm = np.r_[mA, mB], np.r_[wA, wB]
    zA, wzA = gl(-18.0, -3.0, n_z // 3)
    zB, wzB = gl(-3.0, 2.5, 2 * n_z // 3)
    z, wz = np.r_[zA, zB], np.r_[wzA, wzB]
    M2, Z23, Z12 = np.meshgrid(m2, z, z, indexing="ij")
    W = wm[:, None, None] * wz[None, :, None] * wz[None, None, :]
    g23, g12 = np.exp(Z23), np.exp(Z12)
    M3, M1 = M2 - g23, M2 + g12
    logd = (LOG720 - 1.5 * np.log(2 * np.pi) - 0.5 * (M1 ** 2 + M2 ** 2 + M3 ** 2) + 7 * log_ndtr(M3)
            + Z23 + Z12)
    return {"M1": M1.ravel(), "M2": M2.ravel(), "M3": M3.ravel(), "w": (W * np.exp(logd)).ravel()}


def integrate(fam, w, Gd):
    """sum_c P(c) * integral over M2>0 of sigmoid(f_c) (V3: also exact sum over K)."""
    d = FAMS[fam]
    tot = 0.0
    if fam == "V3":
        pi = np.clip((ndtr(Gd["M3"]) - 0.5) / ndtr(Gd["M3"]), 0, 1)
        Ks = [(2, np.where(Gd["M3"] <= 0, 1.0, 0.0))]
        for k in range(3, 11):
            Ks.append((k, np.where(Gd["M3"] > 0, binom.pmf(k - 3, 7, pi), 0.0)))
    else:
        Ks = [(None, np.ones_like(Gd["w"]))]
    n = len(Gd["w"])
    for c in range(3):
        for k, pk in Ks:
            for i in range(0, n, 300_000):                   # chunked to bound memory
                s = slice(i, i + 300_000)
                F = feats(fam, Gd["M1"][s], Gd["M2"][s], Gd["M3"][s], np.full(len(Gd["w"][s]), k if k is not None else 0))
                tot += PC[c] * np.sum(Gd["w"][s] * pk[s] * expit(F @ w[c * d:(c + 1) * d]))
    return tot


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    out = {}
    G = {n: grid(n, n) for n in (64, 96, 128)}
    out["quadrature_mass_nonblind"] = {n: float(g["w"].sum()) for n, g in G.items()}
    out["exact_mass_nonblind"] = 1 - 11 / 1024
    print("mass", out["quadrature_mass_nonblind"], 1 - 11 / 1024, flush=True)
    truths = json.load(open("results/truths.json"))
    data = {}
    for i, name in enumerate(MODELS):
        err, V = make(MODELS[name], 1_000_000, 269100 + i)
        err2, V2 = make(MODELS[name], 1_000_000, 269200 + i)
        data[name] = (err, V, err2, V2)
        print("data", name, err.mean(), flush=True)

    def nb(V):
        return V["M2"] > 0

    def blind_rates(err, V):
        return {k: float(err[(V["M2"] <= 0) & (V["K"] == k)].mean()) for k in (0, 1)}

    def estimate(fam, w, br, Gd):
        return 1 - sum(BLIND[k] * br[k] for k in (0, 1)) - integrate(fam, w, Gd)

    res = {}
    import sys
    for fam in (sys.argv[1:] or list(FAMS)):
        res[fam] = {}
        # (0) own-model fit (trivial pass expected), with quadrature convergence and held-out CV variance ratio
        for name in MODELS:
            err, V, err2, V2 = data[name]
            m = nb(V)
            Vm = {k: v[m] for k, v in V.items()}
            w = logistic(design(fam, Vm), err[m].astype(float))
            br = blind_rates(err, V)
            est = {n: estimate(fam, w, br, g) for n, g in G.items()}
            # control variate on held-out: acc = integral - mean(err - f) ; f = fitted P(err) incl. blind constant
            m2 = nb(V2)
            f = np.empty(len(err2))
            V2m = {k: v[m2] for k, v in V2.items()}
            f[m2] = expit(design(fam, V2m) @ w)
            f[~m2] = np.where(V2["K"][~m2] == 0, br[0], br[1])
            resid = err2 - f
            cv = est[128] - resid.mean()
            truth = ANSWER_KEY_TARGET if name == "target" else truths[name]["acc"]
            res[fam][name] = {"own_fit_est": est, "own_fit_bias": est[128] - truth, "truth": truth,
                              "cv_est": cv, "cv_bias": cv - truth,
                              "var_ratio_cv_vs_sampling": float(resid.var() / err2.var()),
                              "heldout_sample_acc": float(1 - err2.mean()), "n_params": int(len(w) + 2)}
            if name == "target":
                w_target, br_target = w, br
            print(fam, name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res[fam][name].items()
                              if k != "own_fit_est"}, flush=True)
        # (i) K = 0 transfer of the target formula; (iii) refit of 3 per-case intercepts on the scored model
        for name in MODELS:
            if name == "target":
                continue
            err, V, err2, V2 = data[name]
            truth = res[fam][name]["truth"]
            res[fam][name]["transfer_K0_bias"] = estimate(fam, w_target, br_target, G[128]) - truth
            m = nb(V)
            Vm = {k: v[m] for k, v in V.items()}
            X = design(fam, Vm)
            off = X @ w_target
            d = FAMS[fam]
            Xi = np.zeros((len(off), 3))
            for c in range(3):
                Xi[Vm["c"] == c, c] = 1
            b = logistic(Xi, err[m].astype(float), offset=off)
            w3 = w_target.copy()
            for c in range(3):
                w3[c * d] += b[c]
            res[fam][name]["refit_K3_bias"] = estimate(fam, w3, blind_rates(err, V), G[128]) - truth
            print(fam, name, "K0 %.5f  K3 %.5f" % (res[fam][name]["transfer_K0_bias"], res[fam][name]["refit_K3_bias"]),
                  flush=True)
        # (ii) leave-one-out across zoo seeds (fit pooled on the other four, 500k each)
        for s in range(5):
            names = [f"zoo{r}" for r in range(5) if r != s]
            Xs, ys, brs = [], [], []
            for nm in names:
                err, V, _, _ = data[nm]
                sel = np.arange(500_000)
                Vs = {k: v[sel] for k, v in V.items()}
                m = nb(Vs)
                Xs.append(design(fam, {k: v[m] for k, v in Vs.items()}))
                ys.append(err[sel][m].astype(float))
                brs.append(blind_rates(err, V))
            w = logistic(np.concatenate(Xs), np.concatenate(ys))
            br = {k: float(np.mean([b[k] for b in brs])) for k in (0, 1)}
            res[fam][f"zoo{s}"]["loo_bias"] = estimate(fam, w, br, G[128]) - res[fam][f"zoo{s}"]["truth"]
            print(fam, f"zoo{s}", "LOO %.5f" % res[fam][f"zoo{s}"]["loo_bias"], flush=True)
    out["results"] = res
    json.dump(out, open("results/phase1_%s.json" % "_".join(res), "w"), indent=1, default=float)

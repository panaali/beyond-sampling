"""T334: pointwise MC of T317e's form (V1) and two switch-offs: (a) line-set product, (b) tmax cut. Own code.
Transverse constants by my own quadrature (scipy.integrate.quad), checked against cont_e.TABLES."""
import json
import sys
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

phi = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
mu = lambda v: -phi(v) / ndtr(v)
dmu = lambda v: -v * mu(v) + mu(v) ** 2
TMAX = 0.5
REG = {"nb": (0.0, 6.5), "b": (-6.5, 0.0)}


def constants(lo, hi):
    f2 = lambda v: 90 * phi(v) * (1 - ndtr(v)) * ndtr(v) ** 8              # density of v2 (2nd of 10)
    Z = quad(f2, lo, hi, epsabs=1e-13)[0]
    v2 = quad(lambda v: v * f2(v), lo, hi, epsabs=1e-13)[0] / Z
    g12 = quad(lambda v: f2(v) * (phi(v) - v * (1 - ndtr(v))) / (1 - ndtr(v)), lo, hi, epsabs=1e-13)[0] / Z
    g23 = quad(lambda v: f2(v) * quad(lambda u: (ndtr(u) / ndtr(v)) ** 8, -14, v, epsabs=1e-13)[0], lo, hi,
               epsabs=1e-12, limit=200)[0] / Z
    v3 = v2 - g23
    w4 = lambda u: phi(u) * ndtr(u) ** 6                                     # law of v4 = max of 7 below v3
    v4_exact = quad(lambda u: u * w4(u), v3 - 12, v3, epsabs=1e-13)[0] / quad(w4, v3 - 12, v3, epsabs=1e-13)[0]
    # T317e's declared node is the mean on a left-Riemann grid v3-8..v3 (40,000 steps); it differs from the exact
    # mean by ~8e-5. To reproduce the estimator as declared, use the same grid rule (re-derived here).
    u = np.linspace(v3 - 8, v3, 40001)[:-1]; w = w4(u)
    v4 = float((u * w).sum() / w.sum())
    return dict(v4_exact=v4_exact, v2=v2, g12=g12, g23=g23, v3=v3, v4=v4, delta=v3 - v4, m4=mu(v4), dm4=dmu(v4))


K = {r: constants(*REG[r]) for r in REG}


def fwd_ok(model, X, k):
    Wi, Wh, Wo = model
    h = np.zeros((len(X), Wh.shape[0]))
    for t in range(X.shape[1]):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1) == k


def build(P, v1, v2, v3, v4, low):
    """P: (n, 10) argsort positions descending: P[:,0..2] = p1..p3; v4 at the latest free position; lows elsewhere."""
    n = len(P); X = np.empty((n, 10)); r = np.arange(n)
    free = np.sort(P[:, 3:], 1)
    X[r[:, None], free[:, :-1]] = low[:, None]
    X[r, free[:, -1]] = v4
    X[r, P[:, 0]] = v1; X[r, P[:, 1]] = v2; X[r, P[:, 2]] = v3
    return X


def verdicts(model, x):
    n = len(x); P = np.argsort(-x, 1); s = -np.sort(-x, 1)
    g12, g23 = s[:, 0] - s[:, 1], s[:, 1] - s[:, 2]; k = P[:, 1]
    nb = s[:, 1] > 0
    c = {f: np.where(nb, K["nb"][f], K["b"][f]) for f in K["nb"]}
    lin = lambda v4: c["m4"] + c["dm4"] * (v4 - c["v4"])
    def s23(t):
        v3 = c["v2"] - t; v4 = v3 - c["delta"]
        return fwd_ok(model, build(P, c["v2"] + c["g12"], c["v2"], v3, v4, lin(v4)), k)
    def s12(t):
        return fwd_ok(model, build(P, c["v2"] + t, c["v2"], c["v3"], c["v4"] + 0 * t, c["m4"] + 0 * t), k)
    raw23, raw12 = s23(g23), s12(g12)
    raw12 = np.where(nb, raw12, True)                                         # blind rule (kept in all variants)
    in23, in12 = g23 < TMAX, g12 < TMAX
    cut23, cut12 = np.where(in23, raw23, True), np.where(in12, raw12, True)
    v1 = cut23 & cut12
    v3j = c["v2"] - g23; v4j = v3j - c["delta"]
    joint = fwd_ok(model, build(P, c["v2"] + g12, c["v2"], v3j, v4j, lin(v4j)), k)
    prod_off = np.where(nb & in23 & in12, joint, v1)
    tmax_off = raw23 & raw12
    return v1, prod_off, tmax_off


if __name__ == "__main__":
    import cont_e as CE
    for t in CE.TABLES:
        r = "nb" if t["name"].startswith("nb") else "b"
        mine = (K[r]["v2"], K[r]["g12"], K[r]["g23"], K[r]["v4"], K[r]["m4"], K[r]["dm4"])
        theirs = (*t["point"], t["v4"], t["m4"], t["dm4"])
        print(r, np.max(np.abs(np.array(mine) - np.array(theirs))))
        assert np.allclose(mine, theirs, atol=1e-6)
    Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
    truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
    idx = range(len(Z["names"])) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
    N, CH = 200_000, 50_000; out = {}
    for i in idx:
        name = str(Z["names"][i]); m = (Z["Wi"][i].astype(float), Z["Wh"][i].astype(float), Z["Wo"][i].astype(float))
        rng = np.random.default_rng(334000 + i); acc = []
        for _ in range(N // CH):
            acc.append(np.stack(verdicts(m, rng.standard_normal((CH, 10))), 1))
        A = np.concatenate(acc).astype(float)
        e = CE.est(m)[0]
        out[name] = dict(mean=A.mean(0).tolist(), cov=(np.cov(A.T) / N).tolist(), est_e=e, truth=truth[name])
        z = (A[:, 0].mean() - e) / np.sqrt(np.cov(A.T)[0, 0] / N)
        print(name, np.round(A.mean(0), 5), round(e, 5), round(truth[name], 5), "z", round(z, 2), flush=True)
    json.dump(out, open(f"results/pointwise_{idx[0] if len(idx) == 1 else 'all'}.json", "w"))

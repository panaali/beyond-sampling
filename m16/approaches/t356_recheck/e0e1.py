"""T357 step 1: own paired MC of closure c (E0 true v4, E1 one-node v4) on DEV3. usage (repo root):
python m16/approaches/t356_recheck/e0e1.py DEV3_WEIGHTS DEV3_TRUTHS T343_DEV3_JSON -> e0e1.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
T, N = 10, 200_000
mu = lambda v: -norm.pdf(v) / np.maximum(ndtr(v), 1e-300)
# E[max of 7 iid N | < v3] = v3 - int_{-inf}^{v3} (Phi(u)/Phi(v3))^7 du, tabulated on a grid of v3
G3 = np.linspace(-6, 6, 4801)
def _node(v3):
    u = np.linspace(v3 - 12, v3, 24001); f = (ndtr(u) / ndtr(v3)) ** 7
    return v3 - np.trapezoid(f, u)
NODE = np.array([_node(v) for v in G3])
node = lambda v3: np.interp(v3, G3, NODE)


def closure(X, v4):
    n = len(X); o = np.argsort(-X, 1); top = o[:, :3]; Xc = np.empty_like(X)
    free = np.ones((n, T), bool); free[np.arange(n)[:, None], top] = False
    P4 = T - 1 - np.argmax(free[:, ::-1], 1)                       # latest free position
    Xc[:] = mu(v4)[:, None]
    Xc[np.arange(n)[:, None], top] = np.take_along_axis(X, top, 1)
    Xc[np.arange(n), P4] = v4
    return Xc, o[:, 1]


def ok(m, X, y):
    Wi, Wh, Wo = m; h = np.zeros((len(X), len(Wi)))
    for t in range(T):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1) == y


if __name__ == "__main__":
    z = np.load(sys.argv[1]); TR = json.load(open(sys.argv[2])); TR = TR.get("truths", TR); Q = json.load(open(sys.argv[3]))
    out = {}
    for i, n in enumerate(map(str, z["names"])):
        m = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
        X = np.random.default_rng(357 + i).standard_normal((N, T)); S = -np.sort(-X, 1); v3, v4 = S[:, 2], S[:, 3]
        X0, y = closure(X, v4); X1, _ = closure(X, node(v3))
        out[n] = dict(E0=float(ok(m, X0, y).mean()), E1=float(ok(m, X1, y).mean()), truth=TR[n]["acc"], T317f=Q[n]["T317f"]["est"])
    json.dump(out, open(os.path.join(HERE, "e0e1.json"), "w"), indent=1)
    def report(ns, tag):
        t = np.array([out[n]["truth"] for n in ns]); r = {}
        for k in ("E0", "E1", "T317f"):
            e = np.array([out[n][k] for n in ns]) - t; r[k] = dict(bias=100 * e.mean(), sd=100 * e.std(), rms=100 * np.sqrt((e ** 2).mean()))
        d = np.array([out[n]["T317f"] - out[n]["E0"] for n in ns]); c0 = np.array([out[n]["E0"] for n in ns]) - t
        r["T317f-E0"] = dict(mean=100 * d.mean(), sd=100 * d.std(), rms=100 * np.sqrt((d ** 2).mean()), corr_with_E0err=float(np.corrcoef(d, c0)[0, 1]),
                             quadrature_sd=100 * float(np.sqrt(d.var() + c0.var())))
        print(tag, json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in r.items()})); return r
    names = list(out); summ = dict(all100=report(names, "all100"), first10=report(names[:10], "first10"))
    json.dump(summ, open(os.path.join(HERE, "e0e1_summary.json"), "w"), indent=1)


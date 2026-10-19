"""T375 oracle gate (diagnostic MC, never an estimator): best L2 tensor-train fit of the TRUE logits, its argmax
accuracy. Usage: python tt.py MODEL BASIS(H|P) r [r ...]   (README 'T375' pre-registration)."""
import json
import sys
from math import factorial

import numpy as np
from scipy.stats import norm

from run import load_panel, arc_eval

T, NB = 10, 8
KNOTS = norm.ppf(np.arange(1, 7) / 7)


def basis(x, kind):
    """(N, T, NB) per-input basis values."""
    if kind == "H":
        He = [np.ones_like(x), x]
        for k in range(2, NB):
            He.append(x * He[-1] - (k - 1) * He[-2])
        return np.stack([h / np.sqrt(factorial(k)) for k, h in enumerate(He)], -1)
    cols = [np.ones_like(x), x] + [np.maximum(x - c, 0) for c in KNOTS]
    return np.stack(cols, -1)


def predict(G, Ph):
    v = np.ones((len(Ph), 1))
    for t in range(T):
        v = np.einsum("xa,xi,aib->xb", v, Ph[:, t], G[t])
    return v                                    # (N, 10)


def fit(Ph, Y, r, sweeps=4, seed=0):
    rng = np.random.default_rng(seed)
    N = len(Ph)
    ranks = [1] + [r] * (T - 1) + [10]
    G = []
    for t in range(T):
        a, b = ranks[t], ranks[t + 1]
        g = .1 * rng.standard_normal((a, NB, b)); g[:, 0, :] += np.eye(a, b)
        G.append(g)

    def lefts():
        L = [np.ones((N, 1))]
        for t in range(T - 1):
            L.append(np.einsum("xa,xi,aib->xb", L[-1], Ph[:, t], G[t]))
        return L

    def rights():
        R = [None] * T
        R[T - 1] = np.broadcast_to(np.eye(10), (N, 10, 10))
        for t in range(T - 1, 0, -1):
            R[t - 1] = np.einsum("aib,xi,xbo->xao", G[t], Ph[:, t], R[t])
        return R

    def solve(t, L, R):
        a, b = ranks[t], ranks[t + 1]
        P = a * NB * b
        XtX = np.zeros((P, P)); Xty = np.zeros(P)
        for s in range(0, N, 4000):
            Z = np.einsum("xa,xi->xai", L[t][s:s + 4000], Ph[s:s + 4000, t])
            X = np.einsum("xai,xbo->xoaib", Z, R[t][s:s + 4000]).reshape(-1, P)
            XtX += X.T @ X; Xty += X.T @ Y[s:s + 4000].reshape(-1)
        lam = 1e-8 * np.trace(XtX) / P
        return np.linalg.solve(XtX + lam * np.eye(P), Xty).reshape(a, NB, b)

    for sw in range(sweeps):
        R = rights(); L = [np.ones((N, 1))]
        for t in range(T):                      # left to right, left-orthogonalise
            G[t] = solve(t, L, R)
            if t < T - 1:
                a, _, b = G[t].shape
                Q, Rm = np.linalg.qr(G[t].reshape(a * NB, b))
                G[t] = Q.reshape(a, NB, -1); G[t + 1] = np.einsum("cb,bid->cid", Rm, G[t + 1])
                L.append(np.einsum("xa,xi,aib->xb", L[-1], Ph[:, t], G[t]))
        L = lefts(); R = [None] * T
        R[T - 1] = np.broadcast_to(np.eye(10), (N, 10, 10))
        for t in range(T - 1, -1, -1):          # right to left, right-orthogonalise
            G[t] = solve(t, L, R)
            if t > 0:
                a, _, b = G[t].shape
                Q, Rm = np.linalg.qr(G[t].reshape(a, NB * b).T)
                G[t] = Q.T.reshape(-1, NB, b); G[t - 1] = np.einsum("aib,cb->aic", G[t - 1], Rm)
                R[t - 1] = np.einsum("aib,xi,xbo->xao", G[t], Ph[:, t], R[t])
    return G


def rel_l2(P_, Y):
    return float(np.sqrt(((P_ - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()))


if __name__ == "__main__":
    nm, kind, rs = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3:]]
    z, names, truths = load_panel(); i = names.index(nm)
    model = tuple(z[k][i].astype(float) for k in ("Wi", "Wh", "Wo"))
    rng = np.random.default_rng(37500 + i)
    x = rng.standard_normal((100_000, T)); Y = arc_eval.forward(model, x); Ph = basis(x, kind)
    xe = rng.standard_normal((1_000_000, T))
    out = []
    for r in rs:
        G = fit(Ph, Y, r)
        tr = rel_l2(predict(G, Ph), Y)
        ok = 0; se = 0.0; vv = 0.0
        for s in range(0, len(xe), 100_000):
            xc = xe[s:s + 100_000]; yt = arc_eval.forward(model, xc); yp = predict(G, basis(xc, kind))
            ok += (yp.argmax(1) == np.argsort(xc, 1)[:, -2]).sum()
            se += ((yp - yt) ** 2).sum(); vv += ((yt - yt.mean(0)) ** 2).sum()
        res = dict(name=nm, basis=kind, r=r, acc=ok / len(xe), truth=truths[nm]["acc"],
                   gap_pp=100 * (ok / len(xe) - truths[nm]["acc"]), rel_l2_test=float(np.sqrt(se / vv)),
                   rel_l2_train=tr)
        out.append(res); print(json.dumps(res), flush=True)
    json.dump(out, open(f"tt_{nm}_{kind}.json", "w"), indent=1)

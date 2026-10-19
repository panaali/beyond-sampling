"""T402 design (README edit rule). Stages, per round r (round 1 starts at the original weights):
  python design.py grad r     -> results/gE_r{r}.json   g_E = dP(E1(2.0))/dW by line-CMC (2e7 lines; seed 402001 + 10 (r-1))
  python design.py select r   -> results/dir_r{r}.json  S (top-12 |g_E| with |g_E| > 3 s.e.), g_A on S by paired FD, d
  python design.py search r   -> results/search_r{r}.json  eps search (R1): design P (5e6 lines, seed 402003, CRN) and
                                 design accuracy change (plain MC 1e7, seed 402004, CRN)
Start weights of round r: the original model (r = 1) or results/search_r{r-1}.json's best point."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lcmc  # noqa: E402
import rnn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SH = [(16,), (16, 16), (10, 16)]; T0 = 2.0; K = 12; GRID = 0.00025; BUDGET = 0.05     # GRID: amendment R3 (was .0025)
W_ORIG = np.array(json.load(open(os.path.join(HERE, "..", "results", "dirs_orig.json")))["w"])


def J(name, obj=None):
    p = os.path.join(RES, name)
    if obj is None:
        return json.load(open(p))
    json.dump(obj, open(p, "w"), indent=1)


def start(r):
    if r == 1:
        return W_ORIG.copy(), 0.0
    s = J(f"search_r{r - 1}.json"); return np.array(s["best_w"]), s["used_norm"]


def acc_plain(models, N, seed, chunk=200_000):
    """Paired plain-MC accuracies of several models on the same inputs -> (acc array, per-model paired-diff s.e. vs model 0)."""
    rng = np.random.default_rng(seed); n = 0; c = np.zeros(len(models)); d1 = np.zeros(len(models)); d2 = np.zeros(len(models))
    while n < N:
        m = min(chunk, N - n); x = rng.standard_normal((m, 10)); tgt = np.argsort(-x, 1)[:, 1]
        ok = np.array([rnn.forward(mm, x).argmax(1) == tgt for mm in models], float); c += ok.sum(1)
        dd = ok - ok[0]; d1 += dd.sum(1); d2 += (dd ** 2).sum(1); n += m
    mu = d1 / N; return c / N, mu, np.sqrt(np.maximum(d2 / N - mu ** 2, 0) / N)


def grad(r):
    w, _ = start(r); t = time.time(); o = lcmc.e1(rnn.unflat(w, SH), T0, 20_000_000, 402001 + 10 * (r - 1))
    J(f"gE_r{r}.json", dict(round=r, P=o["P"], se=o["se"], grad=o["grad"].tolist(), grad_se=o["grad_se"].tolist(), C_fpe=o["C_fpe"],
                            w=w.tolist(), sec=time.time() - t))
    print("round", r, "P %.4e +- %.1e" % (o["P"], o["se"]), "C %.3g FPE" % o["C_fpe"], "%.0fs" % (time.time() - t))


def select(r):
    G = J(f"gE_r{r}.json"); w = np.array(G["w"]); g = np.array(G["grad"]); s = np.array(G["grad_se"])
    ok = np.flatnonzero(np.abs(g) > 3 * s); S = ok[np.argsort(-np.abs(g[ok]))][:K]
    h = 0.01; models = [rnn.unflat(w, SH)]
    for i in S:
        for sg in (1, -1):
            v = w.copy(); v[i] += sg * h; models.append(rnn.unflat(v, SH))
    a, mu, _ = acc_plain(models, 10_000_000, 402002 + 10 * (r - 1))
    gA = np.array([(a[1 + 2 * k] - a[2 + 2 * k]) / (2 * h) for k in range(len(S))])
    gE = g[S]; dE = -gE / np.linalg.norm(gE); uA = gA / np.linalg.norm(gA)
    d = dE - (dE @ uA) * uA; d /= np.linalg.norm(d)
    J(f"dir_r{r}.json", dict(round=r, S=S.tolist(), gE=gE.tolist(), gE_se=s[S].tolist(), gA=gA.tolist(), acc0=a[0], d=d.tolist(),
                             pred_dlogP_per_eps=float(gE @ d / G["P"]), pred_dacc_per_eps=float(gA @ d), cos_EA=float(gE @ gA / np.linalg.norm(gE) / np.linalg.norm(gA))))
    print("S", S.tolist(), "pred dlogP/deps %.1f  dacc/deps %.4f" % (gE @ d / G["P"], gA @ d))


def hess(r):
    """Amendment R3: accuracy Hessian on S by paired plain MC, and the direction d = -M^-1 g_E (M = -H, clipped)."""
    D = J(f"dir_r{r}.json"); G = J(f"gE_r{r}.json"); w = np.array(G["w"]); S = np.array(D["S"]); k = len(S); h = 0.005
    keys = [(0, 0, 0, 0)]
    for a in range(k):
        keys += [(a, 1, a, 0), (a, -1, a, 0)]
        for b in range(a + 1, k):
            keys += [(a, sa, b, sb) for sa in (1, -1) for sb in (1, -1)]
    models = []
    for a, sa, b, sb in keys:
        v = w.copy()
        if sa:
            v[S[a]] += sa * h
        if sb:
            v[S[b]] += sb * h
        models.append(rnn.unflat(v, SH))
    acc, _, _ = acc_plain(models, 2_000_000, 402005 + 10 * (r - 1)); A = dict(zip(keys, acc)); a0 = A[(0, 0, 0, 0)]
    H = np.zeros((k, k))
    for a in range(k):
        H[a, a] = (A[(a, 1, a, 0)] - 2 * a0 + A[(a, -1, a, 0)]) / h ** 2
        for b in range(a + 1, k):
            H[a, b] = H[b, a] = (A[(a, 1, b, 1)] - A[(a, 1, b, -1)] - A[(a, -1, b, 1)] + A[(a, -1, b, -1)]) / (4 * h * h)
    lam, V = np.linalg.eigh(-H); lc = np.maximum(lam, .05 * lam.max()); Minv = (V / lc) @ V.T
    gE = np.array(D["gE"]); d = -Minv @ gE; d /= np.linalg.norm(d); d_old = np.array(D["d"])
    J(f"dir_r{r}_proj.json", D)
    D.update(d=d.tolist(), H=H.tolist(), eig_negH=lam.tolist(), pred_dlogP_per_eps=float(gE @ d / G["P"]),
             pred_dacc_quad_per_eps2=float(d @ H @ d / 2), old_pred_dacc_quad_per_eps2=float(d_old @ H @ d_old / 2), direction="R3 Hessian")
    J(f"dir_r{r}.json", D)
    print("eig(-H)", np.round(lam, 1), "pred dlogP/deps %.1f; dacc ~ %.1f eps^2 (old direction %.1f eps^2)" % (gE @ d / G["P"], d @ H @ d / 2, d_old @ H @ d_old / 2))


def search(r):
    D = J(f"dir_r{r}.json"); G = J(f"gE_r{r}.json"); w, used = start(r); S = np.array(D["S"]); d = np.array(D["d"])
    P0 = J("gE_r1.json")["P"]; left = BUDGET - used; cache = {}
    def edit(eps):
        v = w.copy(); v[S] += eps * d; return v
    def evaluate(kgrid):
        if kgrid in cache:
            return cache[kgrid]
        eps = kgrid * GRID; v = edit(eps); t = time.time()
        p = lcmc.e1(rnn.unflat(v, SH), T0, 5_000_000, 402003, grad=False)
        a, mu, se = acc_plain([rnn.unflat(W_ORIG, SH), rnn.unflat(v, SH)], 10_000_000, 402004)
        row = dict(k=kgrid, eps=eps, P=p["P"], P_se=p["se"], ratio=p["P"] / P0, dacc=float(mu[1]), dacc_se=float(se[1]),
                   ok=bool(p["P"] <= P0 / 20 and mu[1] >= -2e-4), C_fpe=p["C_fpe"], sec=time.time() - t)
        cache[kgrid] = row; print(row, flush=True)
        J(f"search_r{r}.json", dict(round=r, rows=sorted(cache.values(), key=lambda q: q["k"])))
        return row
    kmax = int(round(left / GRID)); ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= min(kmax, 40)] + [min(kmax, 40)]; hit = None; lo = 0
    for k in ks:
        if evaluate(k)["ok"]:
            hit = k; break
        lo = k
    if hit is not None:
        while hit - lo > 1:                                     # bisection on the grid between lo (fails) and hit (ok)
            mid = (lo + hit) // 2
            if evaluate(mid)["ok"]:
                hit = mid
            else:
                lo = mid
        best = hit
    else:
        good = [q for q in cache.values() if q["dacc"] >= -2e-4]
        best = min(good, key=lambda q: q["P"])["k"] if good else 0
    rows = sorted(cache.values(), key=lambda q: q["k"])
    J(f"search_r{r}.json", dict(round=r, rows=rows, qualified=hit is not None, best_k=best, best_eps=best * GRID,
                                best_w=edit(best * GRID).tolist(), used_norm=used + best * GRID))
    print("round", r, "qualified" if hit is not None else "NOT qualified", "best eps %.4f" % (best * GRID))


if __name__ == "__main__":
    {"grad": grad, "select": select, "hess": hess, "search": search}[sys.argv[1]](int(sys.argv[2]))

"""T411 design (README edit rule). Round r starts at the original weights (r = 1) or the previous round's best point.
  python design2.py grad r    -> results/gW_r{r}.json (g_W, 2e6 v1 lines) and, for r > 1, results/gE_r{r}.json (g_E, 2e7 v3 lines)
  python design2.py hess r    -> results/dir_r{r}.json  S, accuracy Hessian on S, direction d
  python design2.py search r  -> results/search_r{r}.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "repair"))
import design as D402  # noqa: E402
import lcmc  # noqa: E402
import rnn  # noqa: E402
import wline  # noqa: E402

RES = os.path.join(HERE, "results"); SH = D402.SH; T0 = 2.0; GRID = 0.00025; BUDGET = 0.05; KMAX = 200
W_ORIG = D402.W_ORIG


def J(name, obj=None):
    p = os.path.join(RES, name)
    if obj is None:
        return json.load(open(p))
    json.dump(obj, open(p, "w"), indent=1)


def start(r):
    if r == 1:
        return W_ORIG.copy(), 0.0
    s = J(f"search_r{r - 1}.json"); return np.array(s["best_w"]), s["used_norm"]


def grad(r):
    w, _ = start(r); m = rnn.unflat(w, SH); t = time.time(); off = 10 * (r - 1)
    o = wline.w(m, T0, 2_000_000, 411001 + off)
    J(f"gW_r{r}.json", dict(round=r, P=o["P"], se=o["se"], grad=o["grad"].tolist(), grad_se=o["grad_se"].tolist(), C_fpe=o["C_fpe"], w=w.tolist(), sec=time.time() - t))
    print("g_W round", r, "P_W %.4e +- %.1e C %.3g" % (o["P"], o["se"], o["C_fpe"]), flush=True)
    if r == 1:
        G = json.load(open(os.path.join(HERE, "..", "repair", "results", "gE_r1.json"))); J("gE_r1.json", G)
    else:
        e = lcmc.e1(m, T0, 20_000_000, 411002 + off)
        J(f"gE_r{r}.json", dict(round=r, P=e["P"], se=e["se"], grad=e["grad"].tolist(), grad_se=e["grad_se"].tolist(), C_fpe=e["C_fpe"], w=w.tolist()))
        print("g_E round", r, "P_E %.4e +- %.1e" % (e["P"], e["se"]), flush=True)


def hess(r):
    GW, GE = J(f"gW_r{r}.json"), J(f"gE_r{r}.json"); w = np.array(GW["w"])
    gw, sw, ge = np.array(GW["grad"]), np.array(GW["grad_se"]), np.array(GE["grad"])
    okw = np.flatnonzero(np.abs(gw) > 3 * sw); Sw = list(okw[np.argsort(-np.abs(gw[okw]))][:12])
    Se = [i for i in np.argsort(-np.abs(ge)) if i not in Sw][:6] if r == 1 else []
    if r == 1:                                                             # T402's top 6 |g_E| (README rule 1)
        top6 = list(np.argsort(-np.abs(ge))[:6]); Se = [i for i in top6 if i not in Sw]
    else:
        Se = J("dir_r1.json")["S"][len(J("dir_r1.json")["Sw"]):]           # keep round 1's set S
        Sw = J("dir_r1.json")["Sw"]
    S = np.array([int(i) for i in Sw] + [int(i) for i in Se]); k = len(S); h = 0.005
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
    acc, _, _ = D402.acc_plain(models, 1_000_000, 411005 + 10 * (r - 1)); A = dict(zip(keys, acc)); a0 = A[(0, 0, 0, 0)]
    H = np.zeros((k, k))
    for a in range(k):
        H[a, a] = (A[(a, 1, a, 0)] - 2 * a0 + A[(a, -1, a, 0)]) / h ** 2
        for b in range(a + 1, k):
            H[a, b] = H[b, a] = (A[(a, 1, b, 1)] - A[(a, 1, b, -1)] - A[(a, -1, b, 1)] + A[(a, -1, b, -1)]) / (4 * h * h)
    lam, V = np.linalg.eigh(-H); lc = np.maximum(lam, .05 * lam.max()); Minv = (V / lc) @ V.T
    gf = gw[S] / GW["P"] + ge[S] / GE["P"]; d = -Minv @ gf; d /= np.linalg.norm(d)
    J(f"dir_r{r}.json", dict(round=r, S=S.tolist(), Sw=[int(i) for i in Sw], H=H.tolist(), eig_negH=lam.tolist(), d=d.tolist(),
                             pred_dlogPW=float(gw[S] @ d / GW["P"]), pred_dlogPE=float(ge[S] @ d / GE["P"]), pred_dacc_quad=float(d @ H @ d / 2)))
    print("S", S.tolist(), "eig(-H) min/max %.1f/%.1f" % (lam.min(), lam.max()), "pred dlogPW %.1f dlogPE %.1f per eps; dacc %.1f eps^2"
          % (gw[S] @ d / GW["P"], ge[S] @ d / GE["P"], d @ H @ d / 2), flush=True)


def search(r):
    Dd = J(f"dir_r{r}.json"); w, used = start(r); S = np.array(Dd["S"]); d = np.array(Dd["d"])
    PW0, PE0 = J("gW_r1.json")["P"], J("gE_r1.json")["P"]; rows = {}; kmax = min(KMAX, int(round((BUDGET - used) / GRID)))
    def ev(k):
        if k in rows:
            return rows[k]
        v = w.copy(); v[S] += k * GRID * d; m = rnn.unflat(v, SH); t = time.time()
        pw = wline.w(m, T0, 1_000_000, 411003, grad=False); pe = lcmc.e1(m, T0, 5_000_000, 411004, grad=False)
        a, mu, se = D402.acc_plain([rnn.unflat(W_ORIG, SH), m], 10_000_000, 411006)
        q = dict(k=k, eps=k * GRID, PW=pw["P"], PW_se=pw["se"], rW=pw["P"] / PW0, PE=pe["P"], PE_se=pe["se"], rE=pe["P"] / PE0,
                 dacc=float(mu[1]), dacc_se=float(se[1]), Pok=bool(pw["P"] <= PW0 / 10 and pe["P"] <= PE0 / 20), accok=bool(mu[1] >= -1e-4),
                 f=float(np.log(max(pw["P"], 1e-300) / PW0) + np.log(max(pe["P"], 1e-300) / PE0)), sec=time.time() - t)
        rows[k] = q; print({a_: (round(b_, 6) if isinstance(b_, float) else b_) for a_, b_ in q.items()}, flush=True)
        J(f"search_r{r}.json", dict(round=r, rows=sorted(rows.values(), key=lambda z: z["k"])))
        return q
    lo, hit = 0, None
    for k in [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k < kmax] + [kmax]:
        if ev(k)["Pok"]:
            hit = k; break
        lo = k
    if hit is not None:
        while hit - lo > 1:
            mid = (lo + hit) // 2
            if ev(mid)["Pok"]:
                hit = mid
            else:
                lo = mid
    qual = hit is not None and rows[hit]["accok"]
    if qual:
        best = hit
    else:
        feas = [q for q in rows.values() if q["accok"]]; best = min(feas, key=lambda q: q["f"])["k"] if feas else 0
    v = w.copy(); v[S] += best * GRID * d
    J(f"search_r{r}.json", dict(round=r, rows=sorted(rows.values(), key=lambda z: z["k"]), smallest_Pok_k=hit, qualified=qual, best_k=best,
                                best_eps=best * GRID, best_w=v.tolist(), used_norm=used + best * GRID))
    print("round", r, "QUALIFIED" if qual else "not qualified", "best k", best, flush=True)


if __name__ == "__main__":
    {"grad": grad, "hess": hess, "search": search}[sys.argv[1]](int(sys.argv[2]))

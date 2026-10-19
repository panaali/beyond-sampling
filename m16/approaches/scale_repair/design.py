"""T425 design (README, pre-registered rule). Design estimates only (line-CMC for E1, plain MC for accuracy); no truths.
  python design.py <tag>   -> results/<tag>/design.json (log of every stage) and results/<tag>/edit.json (frozen edit)
Stages: t0 scan -> g_E (2e7 lines) -> for K in the ladder: S_K, accuracy Hessian on S_K, d = -M^-1 g_E,S, eps search."""
import hashlib
import json
import os
import sys
import time

import numpy as np

import core
import rnn

T0_GRID = [round(0.25 * i, 2) for i in range(1, 25)]           # 0.25 .. 6.0
GRID_EPS = 0.00025; KMAX = 200                                  # eps = GRID_EPS * k, k <= 200 (L2 <= .05)
_F = float(os.environ.get("T425_DSCALE", 1))                    # smoke tests only (divides every N)
N_SCAN, N_GRAD, N_CAND, N_ACC = (int(n / _F) for n in (2_000_000, 20_000_000, 5_000_000, 10_000_000))
H_STEP = 0.005


def save(path, obj):
    json.dump(obj, open(path, "w"), indent=1, default=float)


def scan_t0(model, tag, log):
    """Smallest-first scan; stop once the design P falls below 1e-9. Pick the in-range t0 nearest 1e-8 in log10."""
    rows = []
    for i, t0 in enumerate(T0_GRID):
        o = core.e1(model, t0, N_SCAN, core.seed(tag, 0) + 100 * i, grad=False); rows.append(dict(t0=t0, P=o["P"], se=o["se"]))
        print("scan", t0, "%.3e +- %.1e" % (o["P"], o["se"]), flush=True)
        if o["P"] < 1e-9:
            break
    log["scan"] = rows
    inr = [r for r in rows if 1e-9 <= r["P"] <= 1e-7]
    pool = inr if inr else [r for r in rows if r["P"] > 0]
    best = min(pool, key=lambda r: abs(np.log10(r["P"]) + 8)); log["t0"] = best["t0"]; log["t0_in_range"] = bool(inr)
    return best["t0"]


def hessian(w, S, shp, N, sd):
    k = len(S); keys = [(0, 0, 0, 0)]
    for a in range(k):
        keys += [(a, 1, a, 0), (a, -1, a, 0)]
        for b in range(a + 1, k):
            keys += [(a, sa, b, sb) for sa in (1, -1) for sb in (1, -1)]
    models = []
    for a, sa, b, sb in keys:
        v = w.copy()
        if sa:
            v[S[a]] += sa * H_STEP
        if sb:
            v[S[b]] += sb * H_STEP
        models.append(rnn.unflat(v, shp))
    acc, _, _ = core.acc_plain(models, N, sd); A = dict(zip(keys, acc)); a0 = A[keys[0]]; Hm = np.zeros((k, k)); h = H_STEP
    for a in range(k):
        Hm[a, a] = (A[(a, 1, a, 0)] - 2 * a0 + A[(a, -1, a, 0)]) / h ** 2
        for b in range(a + 1, k):
            Hm[a, b] = Hm[b, a] = (A[(a, 1, b, 1)] - A[(a, 1, b, -1)] - A[(a, -1, b, 1)] + A[(a, -1, b, -1)]) / (4 * h * h)
    return Hm, len(models) * N


def try_K(K, w, g, gs, P0, t0, shp, tag, log):
    ok = np.flatnonzero(np.abs(g) > 3 * gs); S = ok[np.argsort(-np.abs(g[ok]))][:K]
    rec = dict(K=K, n_resolved=int(len(ok)), S=S.tolist())
    if len(S) < K:
        rec["skipped"] = "fewer than K resolved entries"; log["K"].append(rec); return None
    t = time.time(); Hm, nf = hessian(w, S, shp, int((2_000_000 if K <= 12 else 1_000_000) / _F), core.seed(tag, 3) + K)
    lam, V = np.linalg.eigh(-Hm); lc = np.maximum(lam, .05 * lam.max()); gE = g[S]; d = -(V / lc) @ V.T @ gE; d /= np.linalg.norm(d)
    rec.update(eig_negH=lam.tolist(), d=d.tolist(), pred_dlogP_per_eps=float(gE @ d / P0), hess_fwd=nf, hess_sec=time.time() - t)
    base = rnn.unflat(w, shp); cache = {}; flE = 0.0; nA = 0

    def ev(k):
        nonlocal flE, nA
        if k not in cache:
            v = w.copy(); v[S] += k * GRID_EPS * d; m = rnn.unflat(v, shp)
            p = core.e1(m, t0, N_CAND, core.seed(tag, 4), grad=False); _, mu, se = core.acc_plain([base, m], N_ACC, core.seed(tag, 5))
            flE += p["flops"]; nA += 2 * N_ACC
            cache[k] = dict(k=k, eps=k * GRID_EPS, P=p["P"], P_se=p["se"], ratio=p["P"] / P0, dacc=float(mu[1]), dacc_se=float(se[1]),
                            Pok=bool(p["P"] <= P0 / 20), Aok=bool(mu[1] >= -2e-4))
            print("K", K, cache[k], flush=True)
        return cache[k]
    lo, hi = 0, None                                           # smallest k meeting the P condition (doubling, then bisection)
    for k in [1, 2, 4, 8, 16, 32, 64, 128, KMAX]:
        if ev(k)["Pok"]:
            hi = k; break
        lo = k
    if hi is not None:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ev(mid)["Pok"]:
                hi = mid
            else:
                lo = mid
    rows = sorted(cache.values(), key=lambda q: q["k"]); rec.update(rows=rows, design_E_flops=flE, design_acc_fwd=nA)
    rec["qualified"] = bool(hi is not None and cache[hi]["Aok"]); rec["k_star"] = hi if rec["qualified"] else None
    good = [q for q in rows if q["Aok"] and q["k"] > 0]
    rec["fallback_k"] = min(good, key=lambda q: q["P"])["k"] if good else None
    log["K"].append(rec); return rec


def main(tag):
    out = os.path.join(core.HERE, "results", tag); os.makedirs(out, exist_ok=True)
    H, T, _, _ = core.parse(tag); shp = core.shapes(T, H); model = core.load(tag); w = rnn.flat(model); t = time.time()
    log = dict(tag=tag, H=H, T=T, K=[]); t0 = scan_t0(model, tag, log); save(os.path.join(out, "design.json"), log)
    G = core.e1(model, t0, N_GRAD, core.seed(tag, 1)); g, gs, P0 = G["grad"], G["grad_se"], G["P"]
    log.update(P0=P0, P0_se=G["se"], grad_flops=G["flops"], dlogP_norm=float(np.linalg.norm(g) / P0)); save(os.path.join(out, "design.json"), log)
    print(tag, "t0", t0, "P0 %.3e +- %.1e" % (P0, G["se"]), flush=True)
    recs = {}
    r = try_K(12, w, g, gs, P0, t0, shp, tag, log); recs[12] = r; save(os.path.join(out, "design.json"), log)
    ladder = [6, 3] if (r and r["qualified"]) else [24]
    for K in ladder:
        r = try_K(K, w, g, gs, P0, t0, shp, tag, log); recs[K] = r; save(os.path.join(out, "design.json"), log)
        if not (r and r["qualified"]):
            break
    q = [K for K, r in recs.items() if r and r["qualified"]]
    if q:
        Ks = min(q); kk = recs[Ks]["k_star"]; status = "qualified"
    else:
        cand = [r for r in recs.values() if r and r["fallback_k"]]
        Ks = max(cand, key=lambda r: r["K"])["K"] if cand else None; kk = recs[Ks]["fallback_k"] if Ks else None; status = "fallback"
    log.update(K_star=Ks, k_star=kk, status=status, sec=time.time() - t); save(os.path.join(out, "design.json"), log)
    if Ks is None:
        print(tag, "no edit"); return
    S = np.array(recs[Ks]["S"]); d = np.array(recs[Ks]["d"]); eps = kk * GRID_EPS; v = w.copy(); v[S] += eps * d
    E = dict(tag=tag, H=H, T=T, t0=t0, P0_design=P0, K=int(Ks), S=S.tolist(), eps=eps, status=status, design_row=[q for q in recs[Ks]["rows"] if q["k"] == kk][0],
             w_base=w.tolist(), w_edit=v.tolist())
    E["sha256_w_edit"] = hashlib.sha256(json.dumps(E["w_edit"]).encode()).hexdigest(); save(os.path.join(out, "edit.json"), E)
    print(tag, "frozen", status, "K", Ks, "eps", eps, E["design_row"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

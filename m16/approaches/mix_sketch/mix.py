"""T309 (DIAGNOSTIC SAMPLING): how many step-8 Gaussian mixture components, merged by a readout-aware rule, reproduce
per-unit P(correct) within 10%? The push-forward of each component through steps 8-9 is exact (draws + true network),
an upper bound on the quality of B3's closed-form step.  usage: python mix.py -> results/mix.json"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../oracle_floor")); sys.path.insert(0, os.path.join(HERE, "../mf_verify"))
from panel import dev2  # noqa: E402
from mfv import UNITS, sample as sample_blind, mvn_draw  # noqa: E402

T, K8 = 10, 8
NS, NREF, NPF = 60_000, 200_000, 120_000
LADDER = (1, 2, 4, 8, 16, 32, 50, 64, 128)
TRIPLES = list(itertools.permutations(range(T), 3))


def sample_class(tri, n, rng):
    """N(0,I) given the ordered top-3 positions = tri and x_(2) > 0 (rejection on the sorted values)."""
    out = []
    while sum(len(o) for o in out) < n:
        s = -np.sort(-rng.standard_normal((n, T)), 1); s = s[s[:, 1] > 0]; out.append(s)
    s = np.concatenate(out)[:n]
    x = np.empty((n, T)); rest = [p for p in range(T) if p not in tri]
    for r, p in enumerate(tri):
        x[:, p] = s[:, r]
    perm = np.argsort(rng.random((n, 7)), 1)
    for i, p in enumerate(rest):
        x[:, p] = s[np.arange(n), 3 + perm[:, i]]
    return x


def steps(model, h, xs):
    Wi, Wh, Wo = model; G = []
    for t in range(xs.shape[1]):
        h = np.maximum(h @ Wh.T + xs[:, t:t + 1] * Wi[None], 0); G.append(h > 0)
    return h @ Wo.T, G


def kmeans(F, K, seed=309, iters=15):
    rng = np.random.default_rng(seed); n = len(F)
    C = [F[rng.integers(n)]]; d2 = ((F - C[0]) ** 2).sum(1)
    for _ in range(1, K):                                   # k-means++
        C.append(F[rng.choice(n, p=d2 / d2.sum())]); d2 = np.minimum(d2, ((F - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        D = (F ** 2).sum(1)[:, None] - 2 * F @ C.T + (C ** 2).sum(1)[None]
        lab = D.argmin(1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = F[m].mean(0)
    return lab


def unit_run(model, sampler, j, seed):
    Wi, Wh, Wo = model
    rA, rB, rC = (np.random.default_rng(seed + i) for i in range(3))
    xr = sampler(NREF, rA); L, _ = steps(model, np.zeros((NREF, 16)), xr); P = float((L.argmax(1) == j).mean())
    x = sampler(NS, rB)
    _, _ = None, None
    h = np.zeros((NS, 16))
    for t in range(K8):
        u7 = h @ Wh.T + x[:, t:t + 1] * Wi[None]; h = np.maximum(u7, 0)
    z = np.c_[h, x[:, K8:]]                                  # (NS, 18) state h_8
    zp = np.c_[u7, x[:, K8:]]                                # (NS, 18) pre-activation state, h_8 = relu(u_7) exact
    Lz, G = steps(model, h, x[:, K8:])
    pat = np.c_[G[0], G[1]]; u, cnt = np.unique(pat, axis=0, return_counts=True); g = u[cnt.argmax()]
    G8, G9 = np.diag(g[:16].astype(float)), np.diag(g[16:].astype(float))
    Af = np.c_[Wo @ G9 @ Wh @ G8 @ Wh, Wo @ G9 @ Wh @ G8 @ Wi[:, None], Wo @ G9 @ Wi[:, None]]      # (10,18)
    A = Af[[j]] - np.delete(Af, j, 0)                                                            # (9,18)
    std = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-9)
    f = z @ A.T; u8 = h @ Wh.T + x[:, K8:K8 + 1] * Wi[None]
    feats = {"naive": std(z), "future": std(f), "future_gates": np.c_[std(f), std(u8)],
             "future_gates_X": np.c_[std(f), std(u8)], "pre_future_gates_X": np.c_[std(f), std(u8)]}
    out = {"P": P, "modal_pattern_share": float(cnt.max() / NS)}
    for rule, F in feats.items():
        Z = zp if rule.startswith("pre_") else z
        res = {}
        for K in LADDER:
            lab = kmeans(F, K); Ph = 0.0
            for k in range(K):
                m = lab == k; w = m.mean()
                if w == 0:
                    continue
                zk = Z[m]; nk = max(int(round(NPF * w)), 50)
                if rule.endswith("_X"):          # exact innovations: x_8, x_9 resampled from the component's own draws,
                    xl = zk[:, 16:]; hk = zk[:, :16]; ri = rC.integers(len(zk), size=nk); xd = xl[ri]   # h | x Gaussian
                    if len(zk) > 3:
                        C = np.cov(zk.T); B = C[:16, 16:] @ np.linalg.pinv(C[16:, 16:]); S = C[:16, :16] - B @ C[16:, :16]
                        hd = hk.mean(0) + (xd - xl.mean(0)) @ B.T + mvn_draw(np.zeros(16), S, nk, rC)
                    else:
                        hd = hk[ri]
                    d = np.c_[hd, xd]
                else:
                    d = mvn_draw(zk.mean(0), np.cov(zk.T) if m.sum() > 1 else np.zeros((18, 18)), nk, rC)
                hd = np.maximum(d[:, :16], 0) if rule.startswith("pre_") else d[:, :16]
                Lk, _ = steps(model, hd, d[:, 16:]); Ph += w * (Lk.argmax(1) == j).mean()
            res[K] = float(Ph)
        out[rule] = res
    return out


def passes(Ph, P):
    e = abs(Ph - P); return bool(e <= .1 * min(P, 1 - P) or e <= .005)


if __name__ == "__main__":
    panel = dev2(); MI = (0, 5, 20, 40, 60); rng = np.random.default_rng(309)
    jobs = []
    for mi in MI:
        jobs += [(mi, "blind", int(u)) for u in rng.choice(180, 10, replace=False)[:5]]     # cap: first 5 of each draw
        jobs += [(mi, "class", int(c)) for c in rng.choice(720, 10, replace=False)[:5]]
    path = os.path.join(HERE, "results/mix.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for n, (mi, kind, u) in enumerate(jobs):
        key = f"{mi}:{kind}:{u}"
        if key in res:
            continue
        m = panel[mi][1]
        if kind == "blind":
            un = UNITS[u]; sampler = lambda k, r, un=un: sample_blind(un, k, r); j = un[2]
        else:
            tri = TRIPLES[u]; sampler = lambda k, r, tri=tri: sample_class(tri, k, r); j = tri[1]
        r = unit_run(m, sampler, j, 3090000 + 1000 * mi + 7 * u + (0 if kind == "blind" else 500))
        r.update(model=panel[mi][0], kind=kind, unit=u); res[key] = r
        json.dump(res, open(path, "w"))
        ks = {rule: next((K for K in LADDER if passes(r[rule][K], r["P"])), ">128") for rule in ("naive", "future", "future_gates", "future_gates_X", "pre_future_gates_X")}
        print(n, key, "P %.4f modal %.2f  K*: %s  K1 %.4f K50(fut) %.4f  %.0fs" % (r["P"], r["modal_pattern_share"], ks, r["future"][1], r["future"][50], time.time() - t0), flush=True)

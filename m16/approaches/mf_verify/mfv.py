"""T299 (DIAGNOSTIC SAMPLING): per blind unit, (a) exact Gaussian orthant on exact sampled margin moments and
(b) Gaussian projection of the exact hidden state at step 10-m + exact last m steps (variants G and T).
Own code (no meanfield/ or blind11/ code read).  usage: python mfv.py main|all  -> results/<which>.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import multivariate_normal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../oracle_floor"))
from panel import dev2  # noqa: E402

T = 10
NREF, NMOM, NOR = 400_000, 400_000, 200_000
UNITS = [(q, None, j) for q in range(T) for j in range(T) if j != q] + [(None, a, j) for a in range(T) for j in range(T) if j != a]


def sample(unit, n, rng):
    """Exact draws from N(0,I) | unit. unit = (q, None, j): x_q > 0, others < 0, j = largest negative;
    (None, a, j): all negative, a = largest, j = second largest."""
    q, a, j = unit
    x = np.empty((n, T))
    if q is not None:
        x[:, q] = np.abs(rng.standard_normal(n))
        neg = -np.sort(np.abs(rng.standard_normal((n, 9))), 1)        # descending negatives: neg[:,0] largest
        top = [j]; rest = [p for p in range(T) if p not in (q, j)]
    else:
        neg = -np.sort(np.abs(rng.standard_normal((n, 10))), 1)
        top = [a, j]; rest = [p for p in range(T) if p not in (a, j)]
    for r, p in enumerate(top):
        x[:, p] = neg[:, r]
    perm = np.argsort(rng.random((n, len(rest))), 1)                   # uniform permutation of the remaining roles
    vals = neg[:, len(top):]
    for i, p in enumerate(rest):
        x[:, p] = vals[np.arange(n), perm[:, i]]
    return x


def run(model, x, h=None, t0=0, keep=()):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0])) if h is None else h
    st = {}
    for t in range(t0, T):
        if t in keep:
            st[t] = h.copy()
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T, st


def mvn_draw(mean, cov, n, rng):
    w, V = np.linalg.eigh((cov + cov.T) / 2)
    return mean + (rng.standard_normal((n, len(mean))) * np.sqrt(np.clip(w, 0, None))) @ V.T


def unit_eval(model, unit, seed):
    j = unit[2]
    rA, rB, rC = (np.random.default_rng(seed + i) for i in range(3))
    L, _ = run(model, sample(unit, NREF, rA)); P = float((L.argmax(1) == j).mean())
    x = sample(unit, NMOM, rB)
    L, st = run(model, x, keep=(6, 7, 8, 9))
    marg = L[:, [j]] - np.delete(L, j, 1)                              # (n,9) m_i = L_j - L_i
    mu, C = marg.mean(0), np.cov(marg.T)
    Pa = float(multivariate_normal.cdf(np.zeros(9), mean=-mu, cov=C + 1e-12 * np.eye(9), allow_singular=True, abseps=1e-6, maxpts=200_000))
    # m = 0: Gaussian h_10 -> Gaussian logits (check of (a), by draws)
    Lg = mvn_draw(L.mean(0), np.cov(L.T), NOR, rC); out = {"P": P, "a": Pa, "G0": float((Lg.argmax(1) == j).mean())}
    for m in (1, 2, 3, 4):
        k = T - m
        hk = st[k]; xl = x[:, k:]
        z = np.c_[hk, xl]; zm, zc = z.mean(0), np.cov(z.T)
        d = mvn_draw(zm, zc, NOR, rC)                                    # G: joint Gaussian over (h_k, x_k..x_9)
        xg = np.zeros((NOR, T)); xg[:, k:] = d[:, 16:]
        Lg, _ = run(model, xg, h=d[:, :16], t0=k); out[f"G{m}"] = float((Lg.argmax(1) == j).mean())
        # T: true late inputs, h_k | x_late Gaussian regression with exact sampled moments
        xt = sample(unit, NOR, rC)
        H = slice(0, 16); X = slice(16, 16 + m)
        B = zc[H, X] @ np.linalg.pinv(zc[X, X]); S = zc[H, H] - B @ zc[X, H]
        h0 = zm[H] + (xt[:, k:] - zm[X]) @ B.T + mvn_draw(np.zeros(16), S, NOR, rC)
        Lt, _ = run(model, xt, h=h0, t0=k); out[f"T{m}"] = float((Lt.argmax(1) == j).mean())
    return out


def passes(Ph, P):
    e = abs(Ph - P)
    return bool(e <= .2 * P or (P < .05 and e <= .01))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "all":                      # secondary: smaller samples to fit the 1.5 h cap
        NREF, NMOM, NOR = 100_000, 100_000, 50_000
    panel = dev2(); MI = (0, 5, 20, 40, 60)
    rng = np.random.default_rng(299)
    if which == "main":
        jobs = [(mi, int(u)) for mi in MI for u in rng.choice(180, 10, replace=False)]
    else:
        jobs = [(mi, u) for mi in MI for u in range(180)]
    path = os.path.join(HERE, f"results/{which}.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for n, (mi, u) in enumerate(jobs):
        key = f"{mi}:{u}"
        if key in res:
            continue
        r = unit_eval(panel[mi][1], UNITS[u], 2990000 + 1000 * mi + 3 * u)
        r.update(model=panel[mi][0], unit=list(UNITS[u])); res[key] = r
        if n % 10 == 0 or which == "main":
            json.dump(res, open(path, "w"))
            print(n, key, UNITS[u], "P %.4f a %.4f G0 %.4f T1-4 %s G1-4 %s %.0fs" % (r["P"], r["a"], r["G0"], [round(r[f'T{m}'], 3) for m in (1, 2, 3, 4)],
                  [round(r[f'G{m}'], 3) for m in (1, 2, 3, 4)], time.time() - t0), flush=True)
    json.dump(res, open(path, "w"))
    meths = ["a", "G0"] + [f"T{m}" for m in (1, 2, 3, 4)] + [f"G{m}" for m in (1, 2, 3, 4)]
    summ = {mt: dict(pass_rate=float(np.mean([passes(r[mt], r["P"]) for r in res.values()])),
                     median_abs_err=float(np.median([abs(r[mt] - r["P"]) for r in res.values()]))) for mt in meths}
    summ["n_units"] = len(res)
    json.dump(summ, open(os.path.join(HERE, f"results/{which}_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

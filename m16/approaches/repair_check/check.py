"""T410: independent check of T402's frozen repair + random 12-weight control. Own code (no T402/T364 sampler or
estimator reused). SAMPLING-based (verification truths, not an estimator under test).

  python check.py models                       -> results/models.json   (hash checks, the 52 weight sets; no scores)
  python check.py slab  t0 N seed all|pair     -> results/slab_<t0>_<seed>.npz   conditional law given g23 > t0
  python check.py acc   N seed all|pair        -> results/acc_<seed>.npz         plain N(0, I) inputs
  python check.py plain t0 N seed              -> results/plain_<t0>_<seed>.json sampler check (base model only)

Every job evaluates the listed models on the SAME draws (common random numbers) and stores per-block (1e6-draw) sums,
so ratios and differences get delete-one-block jackknife / batch-means s.e. in summarize.py.

Slab sampler (own derivation, v3 first): with v3 = b, exactly two values above b + t0 and seven below b,
  density of b: G(b) = 360 phi(b) Q(b + t0)^2 Phi(b)^7   (360 = 10! / (2! 7! 1!)),  P(g23 > t0) = int G.
b is drawn from a piecewise-constant proposal q on a fine grid and carries the exact weight w = G(b) / q(b), so the
estimator is unbiased whatever the grid; the two high values are i.i.d. N(0,1) given > b + t0, the seven lows
i.i.d. N(0,1) given < b, and the positions are a uniform random permutation."""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
EDIT_SHA = "6c9ad06a41e82903db7e613d9855279c592a59c276b06d6275b88a6285b9a1f4"   # T402 edit.json sha256_w_edit
NORM, CTRL_SEED, N_RAND, N_SIGN, SIGN_SEED = 0.003, 410001, 40, 10, 410002
BLOCK = 1_000_000


def models():
    """[base, T402, rand_0..rand_39, sign_0..sign_9] as flat 432-vectors (Wi 16 | Wh 16x16 row = to | Wo 10x16)."""
    E = json.load(open(os.path.join(HERE, "t402_edit.json")))
    assert hashlib.sha256(json.dumps(E["w_edit"]).encode()).hexdigest() == EDIT_SHA, "T402 edit hash mismatch"
    wb, we = np.array(E["w_base"]), np.array(E["w_edit"]); S = np.array(E["S"])
    d = we - wb; off = np.setdiff1d(np.arange(432), S)
    assert np.all(d[off] == 0) and abs(np.linalg.norm(d[S]) - NORM) < 1e-9
    W = [wb, we]; names = ["base", "t402"]
    rng = np.random.default_rng(CTRL_SEED)
    for k in range(N_RAND):                      # isotropic direction on the 12 weights of S, L2 norm .003
        z = rng.standard_normal(len(S)); w = wb.copy(); w[S] += NORM * z / np.linalg.norm(z); W.append(w); names.append(f"rand_{k}")
    rng = np.random.default_rng(SIGN_SEED); seen = {tuple(np.ones(len(S), int))}
    while len(names) < 2 + N_RAND + N_SIGN:       # T402's per-weight magnitudes, random signs (T402's own pattern excluded)
        s = tuple(rng.choice([-1, 1], len(S)))
        if s in seen:
            continue
        seen.add(s); w = wb.copy(); w[S] += np.array(s) * d[S]; W.append(w); names.append(f"sign_{len(names) - 2 - N_RAND}")
    return names, W


def split(w):
    return w[:16].copy(), w[16:272].reshape(16, 16).T.copy(), w[272:].reshape(10, 16).T.copy()


def pick(w, x):
    wi, whT, woT = split(w); h = np.zeros((len(x), 16))
    for t in range(x.shape[1]):
        h = h @ whT; h += x[:, t:t + 1] * wi; np.maximum(h, 0, out=h)
    return (h @ woT).argmax(1)


class Slab:
    def __init__(self, t0, lo=-8.0, hi=5.0, n=400_001):
        self.t0 = t0; b = np.linspace(lo, hi, n)
        logG = np.log(360) - b ** 2 / 2 - 0.5 * np.log(2 * np.pi) + 2 * log_ndtr(-(b + t0)) + 7 * log_ndtr(b)
        g = np.exp(logG - logG.max()); m = (g[1:] + g[:-1]) / 2       # proposal: cell mass ~ trapezoid, uniform within cell
        self.cdf = np.concatenate([[0.0], np.cumsum(m)]) / m.sum(); self.b = b; self.dx = b[1] - b[0]
        self.qcell = m / m.sum() / self.dx

    def logG(self, b):
        return np.log(360) - b ** 2 / 2 - 0.5 * np.log(2 * np.pi) + 2 * log_ndtr(-(b + self.t0)) + 7 * log_ndtr(b)

    def draw(self, rng, n):
        c = np.searchsorted(self.cdf, rng.random(n), side="right") - 1; c = np.clip(c, 0, len(self.qcell) - 1)
        b = self.b[c] + rng.random(n) * self.dx
        w = np.exp(self.logG(b)) / self.qcell[c]
        Qc = ndtr(-(b + self.t0))
        hi = -ndtri(rng.random((n, 2)) * Qc[:, None])                      # N given > b + t0
        lo = ndtri(rng.random((n, 7)) * ndtr(b)[:, None])                  # N given < b
        vals = np.column_stack([hi, b, lo]); perm = rng.permuted(np.tile(np.arange(10), (n, 1)), axis=1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
        return x, perm[:, 2], w                                             # position of v3 (rank 3)


def job_slab(t0, N, seed, which):
    names, W = models(); M = len(W) if which == "all" else 2; sl = Slab(t0); rng = np.random.default_rng(seed)
    nb = N // BLOCK; sw = np.zeros(nb); sE = np.zeros((nb, M)); sR1 = np.zeros((nb, M)); cnt = np.zeros((M, 10))
    t = time.time()
    for k in range(nb):
        for _ in range(BLOCK // 200_000):
            x, p3, w = sl.draw(rng, 200_000); top = x.argmax(1); ix = np.arange(len(x))
            sw[k] += w.sum()
            for j in range(M):
                p = pick(W[j], x); sE[k, j] += w[p == p3].sum(); sR1[k, j] += w[p == top].sum()
                cnt[j] += np.bincount((x > x[ix, p][:, None]).sum(1), minlength=10)
        print(f"slab {t0} seed {seed} block {k + 1}/{nb} {time.time() - t:.0f}s", flush=True)
    np.savez(os.path.join(RES, f"slab_{t0}_{seed}.npz"), names=names[:M], sw=sw, sE=sE, sR1=sR1, cnt=cnt, N=N, seed=seed)


def job_acc(N, seed, which):
    names, W = models(); M = len(W) if which == "all" else 2; rng = np.random.default_rng(seed)
    nb = N // BLOCK; sC = np.zeros((nb, M)); t = time.time()
    for k in range(nb):
        for _ in range(BLOCK // 200_000):
            x = rng.standard_normal((200_000, 10)); a = x.argmax(1); y = x.copy(); y[np.arange(len(y)), a] = -np.inf
            tgt = y.argmax(1)
            for j in range(M):
                sC[k, j] += (pick(W[j], x) == tgt).sum()
        if (k + 1) % 10 == 0 or k + 1 == nb:
            print(f"acc seed {seed} block {k + 1}/{nb} {time.time() - t:.0f}s", flush=True)
    np.savez(os.path.join(RES, f"acc_{seed}.npz"), names=names[:M], sC=sC, N=N, seed=seed)


def job_plain(t0, N, seed):
    """Sampler check on the base model only: plain MC of P(g23 > t0) and of E1(t0) vs the slab sampler's."""
    names, W = models(); rng = np.random.default_rng(seed); g = e = 0; n = 0
    while n < N:
        x = rng.standard_normal((500_000, 10)); s = np.sort(x, 1); o = np.argsort(-x, 1)
        m = s[:, -2] - s[:, -3] > t0; g += int(m.sum())
        if m.any():
            e += int((pick(W[0], x[m]) == o[m, 2]).sum())
        n += len(x)
    sl = Slab(t0); rs = np.random.default_rng(seed + 1); ws = []; es = []
    for _ in range(10):
        x, p3, w = sl.draw(rs, 200_000); ws.append(w); es.append(w * (pick(W[0], x) == p3))
    ws = np.concatenate(ws); es = np.concatenate(es)
    out = dict(t0=t0, N=N, seed=seed, plain_pgap=g / N, plain_pgap_se=np.sqrt(g) / N, plain_E1=e / N, plain_E1_se=np.sqrt(e) / N,
               slab_pgap=ws.mean(), slab_pgap_se=ws.std() / np.sqrt(len(ws)), slab_E1=es.mean(), slab_E1_se=es.std() / np.sqrt(len(es)),
               w_cv=ws.std() / ws.mean())
    json.dump(out, open(os.path.join(RES, f"plain_{t0}_{seed}.json"), "w"), indent=1); print(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "models":
        names, W = models()
        json.dump(dict(names=names, sha256_models=hashlib.sha256(np.array(W).tobytes()).hexdigest(),
                       dW={n: (w - W[0]).tolist() for n, w in zip(names, W)}), open(os.path.join(RES, "models.json"), "w"))
        print(len(names), "models")
    elif a[0] == "slab":
        job_slab(float(a[1]), int(float(a[2])), int(a[3]), a[4])
    elif a[0] == "acc":
        job_acc(int(float(a[1])), int(a[2]), a[3])
    elif a[0] == "plain":
        job_plain(float(a[1]), int(float(a[2])), int(a[3]))

"""T319 oracle (DIAGNOSTIC SAMPLING, paired): hard closure verdict 1[m>0] vs smoothed Phi(m / (s ||J||)) for the R0
(top-3 exact) and top-4 closures, on the 80 DEV2 retrains. J from the exact local logit map of the closure point's gate
pattern (forward-mode, equivalent to one adjoint pass per rival).  usage: python sv.py [n=250000]"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../oracle_floor"))
from panel import ROOT, dev2, forward  # noqa: E402

T = 10
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100_000
CH = 50_000


def logits_and_map(model, x):
    Wi, Wh, Wo = model
    n = len(x); h = np.zeros((n, 16)); J = np.zeros((n, 16, T))
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        P = np.einsum("ij,njt->nit", Wh, J); P[:, :, t] += Wi[None]
        g = z > 0; h = np.where(g, z, 0); J = g[:, :, None] * P
    return h @ Wo.T, np.einsum("ki,nit->nkt", Wo, J)


def closure(x, k):
    """Top-k exact, the 10-k lows at E[X | X < v_k]; returns x_c, low mask, s."""
    o = np.argsort(-x, 1); vk = np.take_along_axis(x, o[:, k - 1:k], 1)[:, 0]
    lam = norm.pdf(vk) / np.maximum(ndtr(vk), 1e-300); mu = -lam; s = np.sqrt(np.maximum(1 - vk * lam - lam ** 2, 1e-12))
    low = np.ones_like(x, bool); np.put_along_axis(low, o[:, :k], False, 1)
    xc = np.where(low, mu[:, None], x)
    return xc, low, s, o


def verdicts(model, x, k):
    xc, low, s, o = closure(x, k); j = o[:, 1]; n = len(x); ii = np.arange(n)
    L, A = logits_and_map(model, xc)
    M = L[ii, j][:, None] - L                                    # margins m_i (m_j = 0)
    Jl = (A[ii, j][:, None, :] - A) * low[:, None, :]            # (n,10 rivals,10 inputs), low positions only
    nJ = np.linalg.norm(Jl, axis=2); sd = s[:, None] * nJ
    Mr = M.copy(); Mr[ii, j] = np.inf; ist = Mr.argmin(1); m = M[ii, ist]
    hard = (m > 0).astype(float)
    sm = ndtr(m / np.maximum(sd[ii, ist], 1e-12))
    Z = ndtr(M / np.maximum(sd, 1e-12)); Z[ii, j] = 1.0
    return hard, sm, Z.prod(1)


def region(x):
    s = -np.sort(-x, 1); g12, g23 = s[:, 0] - s[:, 1], s[:, 1] - s[:, 2]
    return np.where(s[:, 1] <= 0, 0, np.where(np.minimum(g12, g23) >= .05, 3, np.where(g23 <= g12, 1, 2)))


if __name__ == "__main__":
    out = {}; t0 = time.time()
    for idx, (name, model) in enumerate(dev2()):
        rng = np.random.default_rng(319000 + idx)
        acc = {f"{c}_{v}": [] for c in ("R0", "top4") for v in ("hard", "smooth", "smoothall")}; regs = []; oks = []
        for _ in range(N // CH):
            x = rng.standard_normal((CH, T)); ok = (forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).astype(float)
            oks.append(ok); regs.append(region(x))
            for c, k in (("R0", 3), ("top4", 4)):
                h, s_, sa = verdicts(model, x, k)
                acc[f"{c}_hard"].append(h); acc[f"{c}_smooth"].append(s_); acc[f"{c}_smoothall"].append(sa)
        ok = np.concatenate(oks); rg = np.concatenate(regs); r = {"acc_sample": float(ok.mean())}
        for key, v in acc.items():
            d = np.concatenate(v) - ok
            r[key] = dict(b=100 * d.mean(), se=100 * d.std() / np.sqrt(len(d)),
                          **{nm: 100 * d[rg == i].sum() / len(d) for i, nm in enumerate(("blind", "band23", "band12", "tail"))})
        out[name] = r
        print(idx, name, " ".join("%s %+.3f" % (k, r[k]["b"]) for k in acc), "%.0fs" % (time.time() - t0), flush=True)
        json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)

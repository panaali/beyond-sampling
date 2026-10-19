"""T403 truth: own v3-conditional Monte Carlo for E1(t0) = P(pick = position of rank 3 AND g23 > t0), x ~ N(0, I_10).
Sampling-based (used only for truths, never by the estimator). Construction: given g23 > t0, the 3rd largest value v3
has density 360 phi(v) Phi(v)^7 (1 - Phi(v + t0))^2 / Z(t0) (Z = P(g23 > t0), quadrature); the two top values are iid
N | > v3 + t0, the seven lows iid N | < v3, and the ten values sit at a uniform random permutation of the positions.
v3 is drawn by inverting a tabulated CDF (401k-point grid on [-8, 8]; model-free). p = Z * hits / n.
Common random numbers: one draw per chunk goes through every model at that t0. Float64 forward (arc_eval.forward's
formula). Per point it also counts hits with g12 = v1 - v2 < .1 (the truth tie share).
usage: python truth.py T0 N SEED MODEL[,MODEL...]  -> results/truth_t{T0}_seed{SEED}.json (checkpointed per chunk)
       python truth.py --plain T0 N SEED MODEL     -> plain unconditional MC (code check)"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as MP  # noqa: E402

T = 10
CHUNK = 500_000


def Z(t0):
    f = lambda v: 360 * np.exp(-v * v / 2) / np.sqrt(2 * np.pi) * ndtr(v) ** 7 * ndtr(-(v + t0)) ** 2
    return quad(f, -12, 12, epsabs=0, epsrel=1e-12, limit=400)[0]


def v3_table(t0):
    v = np.linspace(-8, 8, 400_001)
    lf = -v * v / 2 + 7 * log_ndtr(v) + 2 * log_ndtr(-(v + t0)); f = np.exp(lf - lf.max())
    c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2)]); return v, c / c[-1]


def draw(rng, n, t0, tab):
    v, cdf = tab
    v3 = np.interp(rng.random(n), cdf, v)
    tops = -ndtri(rng.random((n, 2)) * ndtr(-(v3 + t0))[:, None])
    lows = ndtri(rng.random((n, 7)) * ndtr(v3)[:, None])
    vals = np.concatenate([tops, v3[:, None], lows], 1)
    perm = np.argsort(rng.random((n, T)), 1)
    x = np.empty((n, T)); np.put_along_axis(x, perm, vals, 1)
    return x, perm[:, 2], np.abs(tops[:, 0] - tops[:, 1])


def forward(model, x):
    Wi, Wh, Wo = model; h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def run(t0, n, seed, names, check_only=False):
    fn = os.path.join(HERE, "results", f"truth_t{t0:g}_seed{seed}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else dict(t0=t0, seed=seed, Z=Z(t0), n=0, models={})
    for nm in names:
        out["models"].setdefault(nm, dict(hits=0, hits_tie=0, n=0))
    ms = {nm: MP.load(nm, check_only) for nm in names}; tab = v3_table(t0)
    rng = np.random.default_rng([seed, int(round(t0 * 1000))])
    for _ in range(out["n"] // CHUNK):                       # resume: replay the stream to the checkpoint
        draw(rng, CHUNK, t0, tab)
    t = time.time()
    while out["n"] < n:
        x, p3, g12 = draw(rng, CHUNK, t0, tab)
        for nm, m in ms.items():
            r = out["models"][nm]
            if r["n"] > out["n"]:
                continue
            hit = forward(m, x).argmax(1) == p3
            r["hits"] += int(hit.sum()); r["hits_tie"] += int((hit & (g12 < 0.1)).sum()); r["n"] += CHUNK
        out["n"] += CHUNK
        for r in out["models"].values():
            q = r["hits"] / max(r["n"], 1); r["p"] = out["Z"] * q; r["se"] = out["Z"] * np.sqrt(q * (1 - q) / max(r["n"], 1))
        if out["n"] % (20 * CHUNK) == 0 or out["n"] >= n:
            json.dump(out, open(fn, "w"), indent=1)
            print(t0, out["n"], {k: v["hits"] for k, v in out["models"].items()}, "%.0fs" % (time.time() - t), flush=True)
    return out


def plain(t0, n, seed, name):
    m = MP.load(name, check_only=True); rng = np.random.default_rng(seed); hits = 0; cond = 0
    for _ in range(n // CHUNK):
        x = rng.standard_normal((CHUNK, T)); o = np.argsort(x, 1)
        g23 = np.take_along_axis(x, o[:, -2:-1], 1)[:, 0] - np.take_along_axis(x, o[:, -3:-2], 1)[:, 0]
        cond += int((g23 > t0).sum()); hits += int(((forward(m, x).argmax(1) == o[:, -3]) & (g23 > t0)).sum())
    return dict(t0=t0, n=n, hits=hits, p=hits / n, se=np.sqrt(hits) / n, Zhat=cond / n, Z=Z(t0))


if __name__ == "__main__":
    if sys.argv[1] == "--plain":
        r = plain(float(sys.argv[2]), int(float(sys.argv[3])), int(sys.argv[4]), sys.argv[5]); print(r)
        json.dump(r, open(os.path.join(HERE, "results", f"plain_check_{sys.argv[5]}_t{sys.argv[2]}.json"), "w"), indent=1)
    else:
        check = os.environ.get("T403_CHECK") == "1"
        run(float(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3]), sys.argv[4].split(","), check)

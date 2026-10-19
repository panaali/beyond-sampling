"""T425 shared tools (generalised from T402 repair/lcmc.py + design.py to any H, T).
  load(tag)                 -> model (Wi (H,), Wh (H,H), Wo (T,H)) from the T421 registry weights (fetched to data/)
  p_gap(T, t0), draw_e1     -> exact conditional law of x given g23 > t0 (v2 density T(T-1) phi (1-Phi) Phi(v2-t0)^(T-2))
  e1(model, t0, n, seed)    -> line-CMC P(E1(t0)) (+ dP/dW): v3 integrated exactly along its line over (v4, v2 - t0)
  acc_plain(models, N, seed)-> paired plain-MC accuracies (design only; accuracy is not rare)
E1(t0) = {argmax == position of the 3rd largest input, g23 = v2 - v3 > t0}. SAMPLING-CLASS (conditional MC)."""
import os
import subprocess

import numpy as np
from scipy.special import ndtr, ndtri

import lgrad
import rnn
import walk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WDIR = os.path.join(ROOT, "data", "scale_weights")
REG = "origin/claude/t421-scale-study"
GRID = np.linspace(-7.0, 10.0, 2_000_001)
PHI = lambda v: np.exp(-v ** 2 / 2) / np.sqrt(2 * np.pi)


def parse(tag):                                   # m_<H>_<T>_<stream><seed>
    _, H, T, ss = tag.split("_"); return int(H), int(T), ss.rstrip("0123456789"), int(ss.lstrip("abcdefghijklmnopqrstuvwxyz"))


def load(tag):
    p = os.path.join(WDIR, tag + ".npz")
    if not os.path.exists(p):
        os.makedirs(WDIR, exist_ok=True)
        blob = subprocess.run(["git", "-C", ROOT, "show", f"{REG}:m16/approaches/scale_study/weights/{tag}.npz"],
                              check=True, capture_output=True).stdout
        open(p, "wb").write(blob)
    d = np.load(p); return d["Wi"].astype(float), d["Wh"].astype(float), d["Wo"].astype(float)


def shapes(T, H):
    return [(H,), (H, H), (T, H)]


def seed(tag, stage):
    H, T, st, s = parse(tag); return 425_000_000 + H * 100_000 + T * 1000 + (500 if st == "zoo" else 0) + s * 10 + stage


def dens(T, t0):
    return T * (T - 1) * PHI(GRID) * ndtr(-GRID) * ndtr(GRID - t0) ** (T - 2)


def p_gap(T, t0):
    f = dens(T, t0); return float(((f[1:] + f[:-1]) / 2 * np.diff(GRID)).sum())


def cdf(T, t0):
    f = dens(T, t0); c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); return c / c[-1]


def draw_e1(c, T, t0, rng, n):
    v2 = np.interp(rng.random(n), c, GRID); v1 = -ndtri(rng.random(n) * ndtr(-v2))
    v3 = ndtri(ndtr(v2 - t0) * rng.random(n) ** (1 / (T - 2))); lows = ndtri(ndtr(v3)[:, None] * rng.random((n, T - 3)))
    vals = np.column_stack([v1, v2, v3, lows]); perm = np.argsort(rng.random((n, T)), 1)
    x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
    return x, perm, v2, lows.max(1)


def _chunk(model):
    P = sum(int(np.prod(np.shape(w))) for w in model); return int(np.clip(1.2e8 / (8 * P), 500, 20_000))


def e1(model, t0, n, seed_, grad=True):
    T = model[2].shape[0]; c = cdf(T, t0); pg = p_gap(T, t0); rng = np.random.default_rng(seed_); ch = _chunk(model)
    P_ = sum(int(np.prod(np.shape(w))) for w in model); sZ = sZ2 = 0.0; sg = np.zeros(P_); sg2 = np.zeros(P_); fl = 0.0; m0 = 0
    while m0 < n:
        m = min(ch, n - m0); x, perm, v2, v4 = draw_e1(c, T, t0, rng, m); p3 = perm[:, 2]; r = np.arange(m)
        d = np.zeros_like(x); d[r, p3] = 1.0; x0 = x.copy(); x0[r, p3] = 0.0; lo, hi = v4, v2 - t0; nrm = ndtr(hi) - ndtr(lo)
        R, A, B, f = walk.argmax_pieces(model, x0, d, p3, lo, hi); R, A, B = walk.merge(R, A, B); fl += f
        Z = np.zeros(m); np.add.at(Z, R, (ndtr(B) - ndtr(A)) / nrm[R]); sZ += Z.sum(); sZ2 += (Z ** 2).sum()
        if grad:
            G = np.zeros((m, P_))
            for e, keep in ((A, A > lo[R] + 1e-9 * (1 + np.abs(lo[R]))), (B, B < hi[R] - 1e-9 * (1 + np.abs(hi[R])))):
                rr, ee = R[keep], e[keep]
                if len(rr):
                    Gm, _, _, f = lgrad.endpoint_grads(model, x0, d, p3, rr, ee, chunk=ch); fl += f
                    np.add.at(G, rr, (PHI(ee) / nrm[rr])[:, None] * Gm)
            sg += G.sum(0); sg2 += (G ** 2).sum(0)
        m0 += m
    se = lambda s, s2: np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 0) / n)
    out = dict(P=pg * sZ / n, se=pg * se(sZ, sZ2), n=n, flops=fl, p_gap=pg)
    if grad:
        out.update(grad=pg * sg / n, grad_se=pg * se(sg, sg2))
    return out


def acc_plain(models, N, seed_, chunk=100_000):
    """Paired plain-MC accuracies of several models on the same inputs -> (acc, paired diff vs model 0, its s.e.)."""
    T = models[0][2].shape[0]; rng = np.random.default_rng(seed_); n = 0; k = len(models)
    c = np.zeros(k); d1 = np.zeros(k); d2 = np.zeros(k)
    while n < N:
        m = min(chunk, N - n); x = rng.standard_normal((m, T)); tgt = np.argsort(-x, 1)[:, 1]
        ok = np.array([rnn.forward(mm, x).argmax(1) == tgt for mm in models], float); c += ok.sum(1)
        dd = ok - ok[0]; d1 += dd.sum(1); d2 += (dd ** 2).sum(1); n += m
    mu = d1 / N; return c / N, mu, np.sqrt(np.maximum(d2 / N - mu ** 2, 0) / N)


def fpe_flops(T, H):
    """FLOPs of one plain forward pass (the FPE unit for this model)."""
    return T * (2 * H * H + 3 * H) + 2 * T * H

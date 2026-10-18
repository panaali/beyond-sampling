"""T327 paired closure oracle (DIAGNOSTIC: sampled inputs). Independent code (smooth_verdict/ not read).
For each input x: build the closure point xbar from x's own skeleton (R0 / top-4 / R1W), then the verdicts
hard, smooth (max rival, product), (a) cone-clipped (max rival, product), (b) exact 1-D walk along the largest-|J| low.
usage: python oracle.py [n] [models...] -> results/oracle.jsonl (one row per model, resumable)"""
import json
import os
import sys

import numpy as np
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402

T, H = 10, 16
NQ = 64                                         # quantile nodes of N(0,1) | < U for clip moments (model-free)
CLOSURES = ("R0", "top4", "R1W")


def phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def tmean_sd(U):
    lam = phi(U) / np.maximum(ndtr(U), 1e-300)
    return -lam, np.sqrt(np.maximum(1 - U * lam - lam * lam, 1e-12))


def closure_point(x, kind):
    """-> xbar (n,T), free mask (n,T), U (n,), target (n,)"""
    n = len(x); r = np.arange(n); o = np.argsort(-x, 1)
    v3, v4 = x[r, o[:, 2]], x[r, o[:, 3]]
    top = np.zeros((n, T), bool); top[r, o[:, 0]] = top[r, o[:, 1]] = top[r, o[:, 2]] = True
    xb = x.copy(); free = ~top
    if kind == "R0":
        U = v3
    else:
        U = v4
        if kind == "top4":
            p4 = o[:, 3]
        else:                                     # latest position not in the triple
            p4 = np.where(~top, np.arange(T)[None], -1).max(1)
        free[r, p4] = False; xb[r, p4] = v4
    m0, s = tmean_sd(U)
    xb = np.where(free, m0[:, None], xb)
    return xb, free, U, s, o[:, 1]


def fwd_full(model, x):
    """logits (n,T), logit Jacobian A (n,T,T) on x's gate pattern, and per-input gate-cone intervals lo, hi (n,T)."""
    Wi, Wh, Wo = model; n = len(x)
    h = np.zeros((n, H)); J = np.zeros((n, H, T))
    lo = np.full((n, T), -np.inf); hi = np.full((n, T), np.inf)
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi
        Pz = np.matmul(Wh, J); Pz[:, :, t] += Wi
        with np.errstate(divide="ignore", invalid="ignore"):
            d = -z[:, :, None] / Pz                    # step in x_k that zeroes pre-activation (t, i)
        up = np.where(d > 0, d, np.inf).min(1); dn = np.where(d < 0, d, -np.inf).max(1)
        hi = np.minimum(hi, x + up); lo = np.maximum(lo, x + dn)
        g = z > 0
        h = np.where(g, z, 0.0); J = Pz * g[:, :, None]
    return h @ Wo.T, np.matmul(Wo, J), lo, hi


def fwd_dir(model, x, k):
    """logits and their derivative along e_k (k per row), plus per-row distance downward to the next gate flip."""
    Wi, Wh, Wo = model; n = len(x); r = np.arange(n)
    h = np.zeros((n, H)); dh = np.zeros((n, H)); amin = np.full(n, np.inf)
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi
        dz = dh @ Wh.T + (k == t)[:, None] * Wi
        with np.errstate(divide="ignore", invalid="ignore"):
            a = z / dz                                  # moving x_k down by a zeroes z (if a > 0)
        amin = np.minimum(amin, np.where(a > 0, a, np.inf).min(1))
        g = z > 0
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    return h @ Wo.T, dh @ Wo.T, amin


def walk_1d(model, xb, k, U, y, lo_end=-8.0, max_cells=200):
    """Exact P(correct) of the network along x_k in (lo_end, U), law N(0,1)|<U, other inputs at xb. -> P, cells."""
    n = len(xb); r = np.arange(n)
    lam = U - 1e-9 * (1 + np.abs(U)); P = np.zeros(n); cells = np.zeros(n); act = np.ones(n, bool)
    FU = ndtr(U)
    for _ in range(max_cells):
        ii = np.flatnonzero(act)
        if not len(ii):
            break
        x = xb[ii].copy(); x[np.arange(len(ii)), k[ii]] = lam[ii]
        L, dL, amin = fwd_dir(model, x, k[ii]); cells[ii] += 1
        yy = y[ii]; a_lo = np.maximum(lam[ii] - amin, lo_end); a_hi = lam[ii]
        m = L[np.arange(len(ii)), yy][:, None] - L; dm = dL[np.arange(len(ii)), yy][:, None] - dL
        m[np.arange(len(ii)), yy] = 1.0; dm[np.arange(len(ii)), yy] = 0.0
        # m_k(l) = m_k + dm_k (l - lam) > 0 on the segment [a_lo, a_hi]
        with np.errstate(divide="ignore", invalid="ignore"):
            root = lam[ii][:, None] - m / dm
        lb = np.where(dm > 0, root, -np.inf).max(1); ub = np.where(dm < 0, root, np.inf).min(1)
        flat_bad = ((dm == 0) & (m <= 0)).any(1)
        a = np.maximum(lb, a_lo); b = np.minimum(ub, a_hi)
        ok = (b > a) & ~flat_bad
        P[ii] += np.where(ok, (ndtr(b) - ndtr(a)) / FU[ii], 0.0)
        lam[ii] = lam[ii] - amin - 1e-9 * (1 + np.abs(lam[ii]))
        act[ii] = lam[ii] > lo_end
    return P, cells


def verdicts(model, x, kind):
    n = len(x); r = np.arange(n)
    xb, free, U, s, y = closure_point(x, kind)
    L, A, lo, hi = fwd_full(model, xb)
    out = {"hard": (L.argmax(1) == y).astype(float)}
    m = L[r, y][:, None] - L                              # (n,T), m_y = 0
    Jm = (A[r, y][:, None, :] - A) * free[:, None, :]     # (n,T rivals,T inputs), free lows only
    riv = np.ones((n, T), bool); riv[r, y] = False
    rstar = np.where(riv, L, -np.inf).argmax(1)
    # smooth (T319 form)
    sdk = s[:, None] * np.sqrt((Jm ** 2).sum(2)) + 1e-12
    pk = ndtr(m / sdk)
    out["smooth"] = pk[r, rstar]
    out["smooth_prod"] = np.where(riv, pk, 1.0).prod(1)
    # (a) cone-clipped
    u = (np.arange(NQ) + 0.5) / NQ
    nodes = ndtri(ndtr(U)[:, None] * u[None])              # (n,NQ) quantiles of N|<U
    cl = np.clip(nodes[:, None, :], lo[:, :, None], hi[:, :, None])   # (n,T,NQ)
    Ec, Vc = cl.mean(2), cl.var(2)
    Ec = np.where(free, Ec, xb); Vc = np.where(free, Vc, 0.0)
    mu = m + (Jm * (Ec - xb)[:, None, :]).sum(2)
    sd = np.sqrt((Jm ** 2 * Vc[:, None, :]).sum(2)) + 1e-12
    pa = ndtr(mu / sd)
    out["clip"] = pa[r, rstar]
    out["clip_prod"] = np.where(riv, pa, 1.0).prod(1)
    # (b) exact 1-D walk along the largest-|J| free low (max-rival J)
    kstar = np.abs(Jm[r, rstar]).argmax(1)
    P1, cells = walk_1d(model, xb, kstar, U, y)
    out["walk1d"] = P1
    out["_cells"] = cells
    return out


def run_model(model, name, n, chunk=10_000):
    rng = np.random.default_rng(327_000 + sum(map(ord, name)))
    acc = {}; cnt = 0; blind_acc = {}; nb = 0
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T)); y = np.argsort(-x, 1)[:, 1]
        Wi, Wh, Wo = model; h = np.zeros((chunk, H))
        for t in range(T):
            h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
        net = ((h @ Wo.T).argmax(1) == y).astype(float)
        blind = x[np.arange(chunk), y] <= 0
        rows = {"net": net}
        for kind in CLOSURES:
            for k, v in verdicts(model, x, kind).items():
                rows[f"{kind}:{k}"] = v
        for k, v in rows.items():
            acc[k] = acc.get(k, 0.0) + v.sum(); blind_acc[k] = blind_acc.get(k, 0.0) + v[blind].sum()
        cnt += chunk; nb += int(blind.sum())
    return dict(model=name, n=cnt, n_blind=nb, mean={k: v / cnt for k, v in acc.items()},
                blind_sum={k: v / cnt for k, v in blind_acc.items()})


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    ms = cf.dev2_models()
    names = sys.argv[2:] or list(ms)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "oracle.jsonl")
    done = {json.loads(l)["model"] for l in open(path)} if os.path.exists(path) else set()
    with open(path, "a") as f:
        for name in names:
            if name in done:
                continue
            row = run_model(ms[name], name, n)
            f.write(json.dumps(row) + "\n"); f.flush()
            mm = row["mean"]
            print(name, " ".join(f"{k}:{100 * (v - mm['net']):+.2f}" for k, v in mm.items() if k != "net" and "_cells" not in k),
                  " cells:", " ".join(f"{mm[c + ':_cells']:.1f}" for c in CLOSURES), flush=True)


if __name__ == "__main__":
    main()

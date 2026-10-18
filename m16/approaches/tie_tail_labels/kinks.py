"""T272 part (1): kink density of the sci_tao band integrand g(U) = F(face_points(U)) along random straight paths in
the unit cube; and the frozen-gate integrand F0a used by part (2).

walk_sig is core.walk_mass plus a 64-bit hash of the band's combinatorial signature (see README Part 3): for every
cell that carries misclassified mass, the gate flips bounding the cell, the side, and the constraint bounding each
wrong piece (logit pair index, cell boundary flip, tie t=0, or window edge with the binding term of tau).
usage: python kinks.py [npts] [npaths] [cells...]   cells like 16:5 or "orig"  -> results/kinks.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

import core as C
from tail import HERE, get_model

SQ2 = np.sqrt(2.0)
M64 = np.uint64(0x9E3779B97F4A7C15)


def _mix(hsh, code):
    with np.errstate(over="ignore"):
        return (hsh ^ code.astype(np.uint64)) * M64 + np.uint64(0x632BE59BD9B4E019)


def correct_set(L, dL, tc, k, p_lo, p_hi):
    """Correct interval [clo, chi] (within [p_lo, p_hi]) for target k with logits L + (t - tc) dL, numpy first-index
    argmax tie rule; also the indices of the binding constraints (-1 = the piece boundary)."""
    rr = np.arange(len(L))
    c = (L[rr, k][:, None] - L) - (dL[rr, k][:, None] - dL) * tc[:, None]
    e = dL[rr, k][:, None] - dL
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -c / e
    lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
    jj = np.arange(L.shape[1])[None, :]
    bad0 = (e == 0) & ((c < 0) | ((np.abs(c) <= 1e-12 * (1 + np.abs(L).max(1))[:, None]) & (jj < k[:, None])))
    lb[rr, k] = -np.inf; ub[rr, k] = np.inf
    jl, ju = lb.argmax(1), ub.argmin(1)
    lbm, ubm = lb[rr, jl], ub[rr, ju]
    clo = np.maximum(lbm, p_lo); chi = np.minimum(ubm, p_hi)
    ok = (chi > clo) & ~bad0.any(1)
    return clo, chi, ok, np.where(lbm > p_lo, jl, -1), np.where(ubm < p_hi, ju, -1)


def tau_code(kind, y, a, b, T):
    """Which term of the window min binds (0: (o1-u) term, 1: (u-o2) term, 2: cap 8.5)."""
    rows = np.arange(len(y)); u = y[rows, a]
    ys = np.sort(y, 1)
    if kind == "23":
        o1 = ys[:, -1]; o2 = ys[:, -4] if T >= 4 else np.full(len(y), -np.inf)
        t1, t2 = SQ2 * (o1 - u) / 3, SQ2 * (u - o2)
        return np.where(np.minimum(t1, t2) > 8.5, 2, np.where(t1 <= t2, 0, 1))
    o2 = ys[:, -3] if T >= 3 else np.full(len(y), -np.inf)
    return np.where(SQ2 * (u - o2) / 3 > 8.5, 2, 1)


def walk_sig(model, y, a, b, tau, kind, eps=1e-10, max_cells=4000):
    Wi, Wh, Wo = model
    n, T = y.shape; H = Wh.shape[0]; rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    tcode = tau_code(kind, y, a, b, T)
    t = -tau.copy(); mass = np.zeros(n); hsh = np.zeros(n, np.uint64)
    prevflip = np.full(n, -2)                                   # -2 = window edge
    act = np.arange(n); it = 0
    while len(act) and it < max_cells:
        it += 1
        X0, D = y[act], d[act]
        tc = t[act] + eps * (1 + np.abs(t[act]))
        h = np.zeros((len(act), H)); dh = np.zeros_like(h)
        nxt = tau[act].copy(); flip = np.full(len(act), -2)
        for s in range(T):
            z = h @ Wh.T + (X0[:, s] + tc * D[:, s])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, s:s + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc[:, None] - z / dz
            r = np.where((r > tc[:, None]) & np.isfinite(r), r, np.inf)
            jmin = r.argmin(1); rmin = r[np.arange(len(act)), jmin]
            upd = rmin < nxt
            nxt = np.where(upd, rmin, nxt); flip = np.where(upd, s * H + jmin, flip)
            g = z > 0
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        L = h @ Wo.T; dL = dh @ Wo.T
        lo = t[act]; hi = nxt
        for side in (0, 1):
            if side == 0:
                p_lo, p_hi, k = lo, np.minimum(hi, 0.0), tneg[act]
                lo_code = np.where(lo <= -tau[act] + 1e-15, -10 - tcode[act], prevflip[act])
                hi_code = np.where(hi >= 0.0, -5, flip)
            else:
                p_lo, p_hi, k = np.maximum(lo, 0.0), hi, tpos[act]
                lo_code = np.where(lo <= 0.0, -5, prevflip[act])
                hi_code = np.where(hi >= tau[act] - 1e-15, -10 - tcode[act], flip)
            okp = p_hi > p_lo
            if not okp.any():
                continue
            clo, chi, ok, jl, ju = correct_set(L, dL, tc, k, p_lo, p_hi)
            corr = np.where(ok, ndtr(chi) - ndtr(clo), 0.0)
            wrong = np.where(okp, ndtr(p_hi) - ndtr(p_lo) - corr, 0.0)
            mass[act] += wrong
            # signature of the wrong pieces: whole piece / left piece / right piece, with their bounding constraints
            leftw = okp & (~ok | (clo > p_lo)); rightw = okp & (~ok | (chi < p_hi))
            has = leftw | rightw
            if has.any():
                code = ((side * 4 + leftw * 2 + rightw) * 1_000_003 + (lo_code + 20) * 1009 + (hi_code + 20)) * 1_000_033 \
                    + np.where(ok, (jl + 2) * 37 + (ju + 2), 999)
                hs = hsh[act]; hsh[act] = np.where(has, _mix(hs, code), hs)
        prevflip[act] = flip
        t[act] = nxt
        act = act[nxt < tau[act]]
    return mass, hsh


def F0_frozen(model, y, a, b, tau, kind):
    """F0a: misclassified phi-mass on |t| < tau with gates frozen at the face point y (logits exactly linear in t)."""
    Wi, Wh, Wo = model
    n, T = y.shape; H = Wh.shape[0]; rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    h = np.zeros((n, H)); dh = np.zeros_like(h)
    for s in range(T):
        z = h @ Wh.T + y[:, s:s + 1] * Wi[None]
        dz = dh @ Wh.T + d[:, s:s + 1] * Wi[None]
        g = z > 0
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    L = h @ Wo.T; dL = dh @ Wo.T; tc = np.zeros(n)
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    out = np.zeros(n)
    for p_lo, p_hi, k in ((-tau, np.zeros(n), tneg), (np.zeros(n), tau, tpos)):
        clo, chi, ok, _, _ = correct_set(L, dL, tc, k, p_lo, p_hi)
        out += ndtr(p_hi) - ndtr(p_lo) - np.where(ok, ndtr(chi) - ndtr(clo), 0.0)
    return out


def paths(model, kind, npts, npaths, rng):
    T = model[2].shape[0]
    cfg = C.configs(T, kind)
    res = dict(length=0.0, kinks=0, jumps=0, jump_rel=[], frac_band=0.0, Fmean=0.0, kink_F=0.0)
    for _ in range(npaths):
        ci = rng.integers(len(cfg))
        U0 = rng.random(T); v = rng.standard_normal(T); v[0] = 0; v /= np.linalg.norm(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            lo_s = np.where(v > 0, (1e-6 - U0) / v, np.where(v < 0, (1 - 1e-6 - U0) / v, -np.inf))
            hi_s = np.where(v > 0, (1 - 1e-6 - U0) / v, np.where(v < 0, (1e-6 - U0) / v, np.inf))
        s0, s1 = lo_s[1:].max(), hi_s[1:].min()
        s = np.linspace(s0, s1, npts); hstep = s[1] - s[0]
        U = U0[None] + s[:, None] * v[None]; U[:, 0] = (ci + 0.5) / len(cfg)
        y, a, b, tau = C.face_points(T, kind, U)
        F, sig = walk_sig(model, y, a, b, tau, kind)
        ch = np.flatnonzero(sig[1:] != sig[:-1])
        res["length"] += s1 - s0; res["kinks"] += len(ch); res["frac_band"] += (F > 0).mean() / npaths
        res["Fmean"] += F.mean() / npaths
        for i in ch:
            if 1 <= i and i + 2 < npts:
                sl, sr = (F[i] - F[i - 1]) / hstep, (F[i + 2] - F[i + 1]) / hstep
                jump = abs(F[i + 1] - F[i] - 0.5 * (sl + sr) * hstep)
                if jump > 10 * max(abs(sl), abs(sr)) * hstep and jump > 1e-13:
                    res["jumps"] += 1; res["jump_rel"].append(jump / max(F.mean(), 1e-300))
            res["kink_F"] += max(F[i], F[min(i + 1, npts - 1)])
    res["density"] = res["kinks"] / res["length"]
    res["jump_rel_median"] = float(np.median(res["jump_rel"])) if res["jump_rel"] else None
    res["jump_rel_p90"] = float(np.quantile(res["jump_rel"], 0.9)) if res["jump_rel"] else None
    res["jump_rel"] = None
    return res


def main():
    npts, npaths = int(sys.argv[1]), int(sys.argv[2])
    cells = sys.argv[3:]
    fn = os.path.join(HERE, "results", f"kinks_{npts}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for cell in cells:
        if cell in out:
            continue
        t0 = time.time()
        if cell in ("orig", "s005", "s010"):
            model = get_model(cell)
        else:
            H, T, sd = (int(v) for v in cell.split(":"))
            model = C.zoo(H, T, seed=sd)
        rng = np.random.default_rng(272)
        r = {k: paths(model, k, npts, npaths, rng) for k in ("23", "12")}
        T = model[2].shape[0]; _, Z23, Z12 = C.tables(T)
        r["Z"] = [Z23, Z12]; r["sec"] = time.time() - t0
        out[cell] = r
        print(cell, {k: (round(r[k]["density"], 2), r[k]["jumps"], round(r[k]["frac_band"], 3)) for k in ("23", "12")},
              round(r["sec"]), flush=True)
        json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()

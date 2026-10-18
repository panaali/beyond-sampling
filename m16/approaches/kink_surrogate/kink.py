"""T318: kinked-tangent surrogate (Fable r4 D2) and CAP-c hybrids, inside my T312 R1W V1 re-implementation."""
import sys
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/r1w_repro")
import r1w_indep as R  # noqa: E402

T, TRI, TMAX, EPS = R.T, R.TRI, R.TMAX, R.EPS


def _fwd(model, X0, D, g):
    """Value + tangent at x = X0 + g D (gates frozen at that point). Returns z, dz (T, n, H), gates, L, dL."""
    Wi, Wh, Wo = model
    n = len(X0)
    x = X0 + g[:, None] * D
    h = np.zeros((n, Wh.shape[0])); dh = np.zeros_like(h)
    Zs, dZs, Gs = [], [], []
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        dz = dh @ Wh.T + D[:, t:t + 1] * Wi[None]
        on = z > 0
        Zs.append(z); dZs.append(dz); Gs.append(on)
        h = np.where(on, z, 0.0); dh = np.where(on, dz, 0.0)
    return np.array(Zs), np.array(dZs), np.array(Gs), h @ Wo.T, dh @ Wo.T


def walk_cap(model, X0, D, label, cap, tmax=TMAX):
    """Exact walk for at most `cap` cells. Returns intervals, cost (FPE, summed), g reached, finished mask."""
    Wi, Wh, Wo = model
    n = len(X0)
    g = np.zeros(n); alive = np.ones(n, bool); first = np.ones(n, bool); restart = np.zeros(n, int)
    cost = 0.0; ints = [[] for _ in range(n)]; cells = 0
    while alive.any() and cells < cap:
        cells += 1
        idx = np.where(alive)[0]
        ge = g[idx] + EPS
        Z, dZ, G, L, dL = _fwd(model, X0[idx], D[idx], ge)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ge[None, :, None] - Z / dZ
        cand = ((G & (dZ < 0)) | (~G & (dZ > 0))) & (r > ge[None, :, None])
        r = np.where(cand, r, np.inf)
        flat = r.transpose(1, 0, 2).reshape(len(idx), -1)
        k = flat.argmin(1); nxt = flat[np.arange(len(idx)), k]; sflip = k // Wh.shape[0]
        hi = np.minimum(nxt, tmax); lo = g[idx]
        _add_correct(ints, idx, L, dL, ge, label[idx], lo, hi)
        cost += np.where(first[idx], 2.0, 2.0 * (T - restart[idx]) / T).sum()
        first[idx] = False; restart[idx] = sflip
        g[idx] = hi
        alive[idx] = nxt < tmax
    return ints, cost, g, ~alive


def _add_correct(ints, idx, L, dL, ge, lab, lo, hi):
    rr = np.arange(len(idx))
    a = L[rr, lab][:, None] - L; b = dL[rr, lab][:, None] - dL
    a[rr, lab] = np.inf; b[rr, lab] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        root = ge[:, None] - a / b
    Lo = np.maximum(np.where(b > 0, root, -np.inf).max(1), lo)
    Hi = np.minimum(np.where(b < 0, root, np.inf).min(1), hi)
    ok = (Lo < Hi) & ~((b == 0) & (a <= 0)).any(1)
    for k in np.where(ok)[0]:
        ints[idx[k]].append((Lo[k], Hi[k]))


def surrogate(model, X0, D, label, g0, tmax=TMAX, n_riv=2):
    """First-order kink expansion from the cell at g0 (per line). Returns intervals in [g0, tmax] and cost (FPE)."""
    Wi, Wh, Wo = model
    n = len(X0)
    ge = g0 + EPS
    Z, dZ, G, L, dL = _fwd(model, X0, D, ge)
    rr = np.arange(n)
    marg = L[rr, label][:, None] - L
    marg[rr, label] = np.inf
    riv = np.argsort(marg, 1)[:, :n_riv]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ge[None, :, None] - Z / dZ
    cross_on = (~G) & (dZ > 0); cross_off = G & (dZ < 0)
    valid = (cross_on | cross_off) & (r > ge[None, :, None]) & (r < tmax)
    sigma = np.where(cross_on, 1.0, -1.0)
    funcs = []                                        # per rival: (m0, slope, kink locs, kink weights)
    for q in range(n_riv):
        j = riv[:, q]
        c = Wo[label] - Wo[j]                        # (n, H)
        a = c.copy()
        A = np.zeros((T, n, Wh.shape[0]))
        for t in range(T - 1, -1, -1):
            A[t] = a                                  # dm/dh_t
            a = (a * G[t]) @ Wh                       # dm/dh_{t-1}
        w = np.where(valid, sigma * A * dZ, 0.0)      # kink weights (T, n, H)
        m0 = L[rr, label] - L[rr, j]
        s0 = dL[rr, label] - dL[rr, j]
        funcs.append((m0, s0, r, w))
    ints = [[] for _ in range(n)]
    nk = 0
    for i in range(n):
        pieces_breaks = [g0[i], tmax]
        per = []
        for m0, s0, rloc, w in funcs:
            sel = valid[:, i, :]
            locs = rloc[:, i, :][sel]; wts = w[:, i, :][sel]
            o = np.argsort(locs)
            per.append((m0[i], s0[i], locs[o], wts[o]))
            pieces_breaks += list(locs)
            nk += len(locs)
        br = np.unique(np.clip(pieces_breaks, g0[i], tmax))
        for lo_, hi_ in zip(br[:-1], br[1:]):
            if hi_ <= lo_:
                continue
            Lo, Hi = lo_, hi_
            for m0_, s0_, locs, wts in per:
                act = locs <= lo_                       # kinks already passed
                slope = s0_ + wts[act].sum()
                val_lo = m0_ + s0_ * (lo_ - ge[i]) + (wts[act] * (lo_ - locs[act])).sum()
                # margin on [lo_, hi_]: val_lo + slope (g - lo_)
                if slope > 0:
                    Lo = max(Lo, lo_ - val_lo / slope)
                elif slope < 0:
                    Hi = min(Hi, lo_ - val_lo / slope)
                elif val_lo <= 0:
                    Hi = Lo
            if Lo < Hi:
                ints[i].append((Lo, Hi))
    cost = n * (2.0 + 1.0 * n_riv) + nk * 10 / R.FPE_FLOPS
    return ints, cost


def line_sets(method, model, X0, D, label):
    n = len(X0)
    if method == "EXACT":
        ints, cost, _ = R.walk(model, X0, D, label)
        return ints, cost
    if method == "SURR":
        return surrogate(model, X0, D, label, np.zeros(n))
    cap = int(method[-1])                           # CAP1 / CAP2
    ints, cost, g, done = walk_cap(model, X0, D, label, cap)
    idx = np.where(~done)[0]
    if len(idx):
        si, sc = surrogate(model, X0[idx], D[idx], label[idx], g[idx])
        for k, i in enumerate(idx):
            ints[i] += si[k]
        cost += sc
    return ints, cost


def est(model, method, four_lines=False):
    rr = np.arange(len(TRI))
    free = np.array([[p for p in range(T) if p not in t] for t in TRI])
    p4 = free[:, -1]; lows = free[:, :-1]
    flops = 0.0; total = 0.0; sets = {}
    for reg in ("nb", "b"):
        tp = R.TP[reg]
        v2s, g12s, v3s, node = tp["v2"], tp["g12"], tp["v3"], tp["node"]
        delta = v3s - node; m0, dm0 = R.mu(node), R.dmu(node)
        X0 = np.zeros((len(TRI), T)); D = np.zeros((len(TRI), T))
        X0[rr, TRI[:, 0]] = v2s + g12s; X0[rr, TRI[:, 1]] = v2s
        X0[rr, TRI[:, 2]] = v2s; D[rr, TRI[:, 2]] = -1.0
        X0[rr, p4] = v2s - delta; D[rr, p4] = -1.0
        X0[rr[:, None], lows] = m0 + dm0 * (v2s - delta - node); D[rr[:, None], lows] = -dm0
        i23, c23 = line_sets(method, model, X0, D, TRI[:, 1]); flops += c23 * R.FPE_FLOPS
        sets[reg + "23"] = i23
        walk12 = reg == "nb" or four_lines
        if walk12:
            X0 = np.zeros((len(TRI), T)); D = np.zeros((len(TRI), T))
            X0[rr, TRI[:, 0]] = v2s; D[rr, TRI[:, 0]] = 1.0
            X0[rr, TRI[:, 1]] = v2s; X0[rr, TRI[:, 2]] = v3s; X0[rr, p4] = node
            X0[rr[:, None], lows] = m0
            i12, c12 = line_sets(method, model, X0, D, TRI[:, 1]); flops += c12 * R.FPE_FLOPS
            sets[reg + "12"] = i12
        x, w = R.GL[reg]
        dens = 720 * norm.pdf(x)
        for k in range(len(TRI)):
            s23 = R._merge(i23[k] + [(TMAX, np.inf)])
            A23 = sum((ndtr(x - a) ** 8 - (ndtr(x - b) ** 8 if np.isfinite(b) else 0.0)) / 8 for a, b in s23)
            if walk12:
                s12 = R._merge(i12[k] + [(TMAX, np.inf)])
                A12 = sum((1.0 if not np.isfinite(b) else ndtr(x + b)) - ndtr(x + a) for a, b in s12)
            else:
                A12 = 1 - ndtr(x)
            total += float((w * dens * A12 * A23).sum())
    return total / len(TRI), flops, sets


def measure(iv):
    return sum(b - a for a, b in R._merge([(max(a, 0), min(b, TMAX)) for a, b in iv if b > a]))


def mismatch(s1, s2):
    """Symmetric-difference measure on [0, tmax] of two interval lists."""
    pts = sorted(set([0, TMAX] + [p for a, b in s1 + s2 for p in (a, b) if 0 <= p <= TMAX]))
    tot = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        mid = (a + b) / 2
        in1 = any(x <= mid < y for x, y in s1); in2 = any(x <= mid < y for x, y in s2)
        tot += (b - a) * (in1 != in2)
    return tot

"""T320: my own instrumented R1W V1 / T317c-style estimator (G = 20) with the four savings EXECUTED:
continuity restarts, B2 prefix tree for the first cell, merging touching correct pieces, model-free Phi tables.
FLOPs are tallied per executed array op on weight-dependent data (multiply-add 2, other scalar op 1, Phi 20).
Built on ../r1w_repro/r1w_indep.py (T312); no code from r1_cont is used."""
import sys
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/r1w_repro")
import r1w_indep as R  # noqa: E402

T, TRI, TMAX, EPS = R.T, R.TRI, R.TMAX, R.EPS
G_NODES = 20
PHI = 20
_t, _w = np.polynomial.legendre.leggauss(G_NODES)
GL = {"nb": (3.25 + 3.25 * _t, 3.25 * _w), "b": (-3.25 + 3.25 * _t, 3.25 * _w)}
# ---- model-free tables (listed; identical for every model; precomputed once) ----
TABLES = {}
for reg, (x, w) in GL.items():
    TABLES[reg] = dict(dw8=w * 720 * norm.pdf(x) / 8.0,        # weight x density, with the 1/8 of the g23 profile
                       phi8_0=ndtr(x) ** 8,                       # g23 endpoint 0
                       phi8_tmax=ndtr(x - TMAX) ** 8,             # g23 endpoint tmax
                       phi_0=ndtr(x),                             # g12 endpoint 0
                       phi_tmax=ndtr(x + TMAX),                   # g12 endpoint tmax
                       blind_A12=1 - ndtr(x))                     # blind g12 (assumed correct)


class Count:
    def __init__(self):
        self.n = 0.0
        self.by = {}

    def add(self, k, tag):
        self.n += k
        self.by[tag] = self.by.get(tag, 0.0) + k


ROOT_RULE = "direction"          # "direction" (pre-registered run, 11H) or "ahead" (v2, post hoc: 6H)


def _roots(z, dz, ge, cnt, tag):
    if ROOT_RULE == "ahead":
        m, H = z.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ge[:, None] - z / dz                                     # 2 / elem
        r = np.where((r > ge[:, None]) & np.isfinite(r), r, np.inf)      # 3 / elem
        k = r.argmin(1)                                                  # 1 / elem
        cnt.add(m * H * 6, tag)
        return r[np.arange(m), k], k
    return _roots_dir(z, dz, ge, cnt, tag)


def _roots_dir(z, dz, ge, cnt, tag):
    """Per-row min root ahead and its neuron index for one step. z, dz: (m, H)."""
    m, H = z.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ge[:, None] - z / dz                     # 2 ops / elem
    cand = (((z > 0) & (dz < 0)) | ((z <= 0) & (dz > 0))) & (r > ge[:, None])   # 7 ops / elem
    r = np.where(cand, r, np.inf)                    # 1
    k = r.argmin(1)                                  # 1
    cnt.add(m * H * 11, tag)
    return r[np.arange(m), k], k


def first_cell(model, X0, D, cnt):
    """B2: value, tangent, root minima and logits computed once per distinct (input, tangent) prefix."""
    Wi, Wh, Wo = model
    n, H = len(X0), Wh.shape[0]
    ge = np.full(n, EPS)
    x = X0 + EPS * D                                                  # model-free
    Z = np.zeros((n, T, H)); dZ = np.zeros((n, T, H)); Gt = np.zeros((n, T, H), bool)
    rmin = np.zeros((n, T)); rarg = np.zeros((n, T), int)
    gid = np.zeros(n, int); hU = np.zeros((1, H)); dhU = np.zeros((1, H))
    for t in range(T):
        key = np.c_[gid, x[:, t], D[:, t]]
        uk, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        par = gid[first_idx]
        u = len(uk)
        z = hU[par] @ Wh.T + x[first_idx, t][:, None] * Wi[None]
        dz = dhU[par] @ Wh.T + D[first_idx, t][:, None] * Wi[None]
        cnt.add(u * 2 * (2 * H * H + 2 * H), "b2_value_tangent")
        on = z > 0
        hU = np.where(on, z, 0.0); dhU = np.where(on, dz, 0.0)
        cnt.add(u * 3 * H, "b2_gates")
        rm, ra = _roots(z, dz, np.full(u, EPS), cnt, "b2_roots")
        Z[:, t] = z[inv]; dZ[:, t] = dz[inv]; Gt[:, t] = on[inv]; rmin[:, t] = rm[inv]; rarg[:, t] = ra[inv]
        gid = inv
    L = hU @ Wo.T; dL = dhU @ Wo.T
    cnt.add(len(hU) * 2 * 2 * Wo.size, "b2_logits")
    return Z, dZ, Gt, rmin, rarg, L[gid], dL[gid]


def correct_piece(L, dL, g, lab, lo, hi, cnt):
    m = len(L); r = np.arange(m)
    a = L[r, lab][:, None] - L; b = dL[r, lab][:, None] - dL                    # 20
    a[r, lab] = np.inf; b[r, lab] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        root = g[:, None] - a / b                                              # 20
    Lo = np.maximum(np.where(b > 0, root, -np.inf).max(1), lo)                 # 10+10+10+1
    Hi = np.minimum(np.where(b < 0, root, np.inf).min(1), hi)                  # 31
    ok = (Lo < Hi) & ~((b == 0) & (a <= 0)).any(1)                             # 1+1+10+10+10+10+1
    cnt.add(m * 145, "correct_interval")
    return Lo, Hi, ok


def walk_cont(model, X0, D, lab, cnt):
    Wi, Wh, Wo = model
    n, H = len(X0), Wh.shape[0]
    Z, dZ, Gt, rmin, rarg, L, dL = first_cell(model, X0, D, cnt)
    g = np.zeros(n)
    pieces = [[] for _ in range(n)]
    alive = np.arange(n)
    while len(alive):
        nxt = rmin[alive].min(1); s = rmin[alive].argmin(1)
        cnt.add(len(alive) * T, "next_event")
        hi = np.minimum(nxt, TMAX)
        Lo, Hi, ok = correct_piece(L[alive], dL[alive], g[alive], lab[alive], g[alive], hi, cnt)
        for k in np.where(ok)[0]:
            pieces[alive[k]].append((Lo[k], Hi[k]))
        cont = nxt < TMAX
        idx = alive[cont]; s = s[cont]; nx = nxt[cont]
        if not len(idx):
            break
        j = rarg[idx, s]
        dlt = nx - g[idx]
        Z[idx] += dZ[idx] * dlt[:, None, None]; L[idx] += dL[idx] * dlt[:, None]   # affine continuation
        cnt.add(len(idx) * (2 * T * H + 2 * 10), "continuation")
        g[idx] = nx
        Gt[idx, s, j] = ~Gt[idx, s, j]
        cnt.add(len(idx), "toggle")
        for sv in np.unique(s):                          # group by restart step: recompute only steps > s*
            ii = idx[s == sv]
            dh = np.where(Gt[ii, sv], dZ[ii, sv], 0.0)
            cnt.add(len(ii) * H, "tangent_recompute")
            for t in range(sv + 1, T):
                dZ[ii, t] = dh @ Wh.T + D[ii, t][:, None] * Wi[None]
                dh = np.where(Gt[ii, t], dZ[ii, t], 0.0)
                cnt.add(len(ii) * (2 * H * H + 2 * H + H), "tangent_recompute")
            dL[ii] = dh @ Wo.T
            cnt.add(len(ii) * 2 * Wo.size, "logit_tangent")
            for t in range(sv, T):                       # roots for steps >= s*
                rm, ra = _roots(Z[ii, t], dZ[ii, t], g[ii] + EPS, cnt, "restart_roots")
                rmin[ii, t] = rm; rarg[ii, t] = ra
        alive = idx
    return pieces


def _merge(iv, cnt):
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    cnt.add(len(iv), "merge")
    return out


def profile(pieces, reg, kind, cnt):
    """A(v2) at the GL nodes for one line from its correct pieces + the model-free tail [tmax, inf)."""
    x, _ = GL[reg]; tb = TABLES[reg]
    iv = _merge(pieces + [(TMAX, np.inf)], cnt)
    A = np.zeros(G_NODES)
    for a, b in iv:
        if kind == "23":
            fa = tb["phi8_0"] if a == 0 else (tb["phi8_tmax"] if a == TMAX else None)
            if fa is None:
                fa = ndtr(x - a) ** 8; cnt.add(G_NODES * (1 + PHI + 3), "phi")
            fb = 0.0 if not np.isfinite(b) else (tb["phi8_tmax"] if b == TMAX else None)
            if fb is None:
                fb = ndtr(x - b) ** 8; cnt.add(G_NODES * (1 + PHI + 3), "phi")
            A += fa - fb
        else:
            fa = tb["phi_0"] if a == 0 else (tb["phi_tmax"] if a == TMAX else None)
            if fa is None:
                fa = ndtr(x + a); cnt.add(G_NODES * (1 + PHI), "phi")
            fb = 1.0 if not np.isfinite(b) else (tb["phi_tmax"] if b == TMAX else None)
            if fb is None:
                fb = ndtr(x + b); cnt.add(G_NODES * (1 + PHI), "phi")
            A += fb - fa
        cnt.add(G_NODES * 2, "profile_sum")
    return A


def est(model, budget=None, return_count=False):
    cnt = Count()
    rr = np.arange(len(TRI))
    free = np.array([[p for p in range(T) if p not in t] for t in TRI])
    p4 = free[:, -1]; lows = free[:, :-1]
    total = 0.0
    for reg in ("nb", "b"):
        tp = R.TP[reg]
        v2s, g12s, v3s, node = tp["v2"], tp["g12"], tp["v3"], tp["node"]
        delta = v3s - node; m0, dm0 = R.mu(node), R.dmu(node)
        X0 = np.zeros((len(TRI), T)); D = np.zeros((len(TRI), T))
        X0[rr, TRI[:, 0]] = v2s + g12s; X0[rr, TRI[:, 1]] = v2s
        X0[rr, TRI[:, 2]] = v2s; D[rr, TRI[:, 2]] = -1.0
        X0[rr, p4] = v2s - delta; D[rr, p4] = -1.0
        X0[rr[:, None], lows] = m0 + dm0 * (v2s - delta - node); D[rr[:, None], lows] = -dm0
        p23 = walk_cont(model, X0, D, TRI[:, 1], cnt)
        if reg == "nb":
            X0 = np.zeros((len(TRI), T)); D = np.zeros((len(TRI), T))
            X0[rr, TRI[:, 0]] = v2s; D[rr, TRI[:, 0]] = 1.0
            X0[rr, TRI[:, 1]] = v2s; X0[rr, TRI[:, 2]] = v3s; X0[rr, p4] = node
            X0[rr[:, None], lows] = m0
            p12 = walk_cont(model, X0, D, TRI[:, 1], cnt)
        dw8 = TABLES[reg]["dw8"]
        for k in range(len(TRI)):
            A23 = profile(p23[k], reg, "23", cnt)
            A12 = profile(p12[k], reg, "12", cnt) if reg == "nb" else TABLES[reg]["blind_A12"]
            total += float((A12 * A23 * dw8).sum())
            cnt.add(G_NODES * 3 + 1, "combine")
    acc = total / len(TRI)
    return (acc, cnt.n, cnt) if return_count else (acc, cnt.n)

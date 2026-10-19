"""T451: own exact line walk for the frozen T10 line set (independent of frozen/walk.py and of T449's sens.py).

Algorithm: grid refinement through time. Along x(s) = w + s d the hidden state h_t(s) is continuous and piecewise linear
in s. Keep it as values on a sorted grid of s that contains every breakpoint. At step t evaluate z_t = Wh h_{t-1} + x_t Wi
at the grid points; inside an interval z_t is linear, so each strict sign change of a unit gives an exact root that is
inserted (z at the new point by linear interpolation), then h_t = relu(z_t). After T steps the logits are linear on each
interval and the correct set of an interval is an intersection of half-lines.

Model-free geometry (face points, windows, strata, Sobol rule) comes from the frozen T10 files, hashes asserted.
"""
import hashlib
import os
import sys

import numpy as np
from numba import njit
from scipy.special import stdtr

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
SHA = {"core.py": "48dbba49ab13", "estimator.py": "fb7902e12240", "hom.py": "a91f019fca21", "walk.py": "a38258f586b4"}
for f, h in SHA.items():
    assert hashlib.sha256(open(os.path.join(FROZEN, f), "rb").read()).hexdigest()[:12] == h, f
sys.path.insert(0, FROZEN)
import core as C  # noqa: E402
import estimator as E  # noqa: E402

T = 10
RULE_SEED = E.RULE_SEED
STRATA = E.strata(T)
DATA = os.path.join(HERE, "../../../data/dev3")
MODELS = [f"dev3_s50{i}" for i in range(5)]
ERR = {64_000: 1.0e-3, 1_000_000: 2.0e-4, 4_000_000: 9.0e-5}   # T429 DEV3-100 RMS


def load(name):
    assert hashlib.sha256(open(os.path.join(DATA, "weights.npz"), "rb").read()).hexdigest()[:8] == "62c95203"
    z = np.load(os.path.join(DATA, "weights.npz"))
    i = list(z["names"]).index(name)
    return tuple(np.ascontiguousarray(z[k][i], dtype=np.float64) for k in ("Wi", "Wh", "Wo"))


def kcdf(s):
    """Student-t_{T-1} mass of s sqrt(T-1) (the radial weight of the frozen T10 recipe)."""
    nu = T - 1
    return stdtr(nu, np.clip(s, -1e300, 1e300) * np.sqrt(nu))


def lines(j, U):
    """Line data of stratum j for uniform nodes U: unit face direction w, d, window [-tw, tw], targets, stratum
    weight, and the model-free crease labels (index of the largest low input, active window term)."""
    kind, h = STRATA[j]
    y, a, b, tau, wt = E.face_points_strat(T, kind, h, U)
    r = np.linalg.norm(y, axis=1)
    w = y / r[:, None]
    tw = np.minimum(tau / r, 1e8)
    n = len(U); rows = np.arange(n)
    d = np.zeros((n, T)); d[rows, a] = 1 / np.sqrt(2); d[rows, b] = -1 / np.sqrt(2)
    kpos = a if kind == "23" else b
    kneg = b if kind == "23" else a
    # model-free creases: which low is the max, which window term is active
    nlow = T - 3 if kind == "23" else T - 2
    col = 3 if kind == "23" else 2
    lab = np.argmax(U[:, col:col + nlow], axis=1).astype(np.int64)
    if kind == "23":
        cfg = C.configs(T, kind)
        p1 = cfg[np.minimum((U[:, 0] * len(cfg)).astype(int), len(cfg) - 1), 2]
        u, o1 = y[rows, a], y[rows, p1]
        lows = y.copy(); lows[rows, a] = -np.inf; lows[rows, b] = -np.inf; lows[rows, p1] = -np.inf
        lab = lab * 2 + ((o1 - u) / 3 < (u - lows.max(1))).astype(np.int64)
    return w, d, tw, kneg.astype(np.int64), kpos.astype(np.int64), wt, lab


@njit(cache=True)
def _forward_pattern(Wi, Wh, x):
    """Gate pattern (T, H) at input x and the input-space Jacobians: J[t] = d h_t / d x (H, T) and
    G[t] = d z_t / d x (H, T)."""
    H = Wh.shape[0]; TT = x.shape[0]
    h = np.zeros(H); J = np.zeros((H, TT))
    pat = np.zeros((TT, H), np.bool_)
    G = np.zeros((TT, H, TT))
    for t in range(TT):
        z = Wh @ h + x[t] * Wi
        Gz = Wh @ J
        for i in range(H):
            Gz[i, t] += Wi[i]
        G[t] = Gz
        for i in range(H):
            on = z[i] > 0
            pat[t, i] = on
            h[i] = z[i] if on else 0.0
            for c in range(TT):
                J[i, c] = Gz[i, c] if on else 0.0
    return pat, J, G


@njit(cache=True)
def _pat_hash(pat):
    hsh = np.uint64(1469598103934665603)
    for t in range(pat.shape[0]):
        for i in range(pat.shape[1]):
            hsh = (hsh ^ np.uint64(pat[t, i] + 1)) * np.uint64(1099511628211)
    return hsh


@njit(cache=True)
def walk_line(Wi, Wh, Wo, w, d, tw, kneg, kpos, want_gates, maxg=20000, maxp=512):
    """Returns: wrong pieces (P, 2), decision roots (R, 4: s, k, j, side), gate roots (Q, 3: s, t, i), status."""
    H = Wh.shape[0]; TT = w.shape[0]; O = Wo.shape[0]
    s = np.empty(maxg); hv = np.empty((maxg, H))
    s[0] = -tw; s[1] = 0.0; s[2] = tw; G = 3
    for g in range(3):
        for i in range(H):
            hv[g, i] = 0.0
    gates = np.empty((maxp * 8, 3)); ng = 0
    zs = np.empty((maxg, H)); s2 = np.empty(maxg); hv2 = np.empty((maxg, H))
    for t in range(TT):
        for g in range(G):
            xt = w[t] + s[g] * d[t]
            for i in range(H):
                acc = xt * Wi[i]
                for k in range(H):
                    acc += Wh[i, k] * hv[g, k]
                zs[g, i] = acc
        # merge roots into the grid
        G2 = 0
        rts = np.empty(H); ui = np.empty(H, np.int64)
        for g in range(G):
            if G2 >= maxg - H - 1:
                return np.zeros((0, 2)), np.zeros((0, 4)), np.zeros((0, 3)), 1
            s2[G2] = s[g]
            for i in range(H):
                hv2[G2, i] = zs[g, i]
            G2 += 1
            if g == G - 1:
                break
            nr = 0
            for i in range(H):
                za = zs[g, i]; zb = zs[g + 1, i]
                if (za > 0 and zb < 0) or (za < 0 and zb > 0):
                    rts[nr] = s[g] + (s[g + 1] - s[g]) * (za / (za - zb)); ui[nr] = i; nr += 1
            if nr == 0:
                continue
            order = np.argsort(rts[:nr])
            for q in range(nr):
                r = rts[order[q]]
                f = (r - s[g]) / (s[g + 1] - s[g])
                s2[G2] = r
                for i in range(H):
                    hv2[G2, i] = zs[g, i] + f * (zs[g + 1, i] - zs[g, i])
                hv2[G2, ui[order[q]]] = 0.0          # exact zero at its own root
                G2 += 1
                if want_gates and ng < gates.shape[0]:
                    gates[ng, 0] = r; gates[ng, 1] = t; gates[ng, 2] = ui[order[q]]; ng += 1
        G = G2
        for g in range(G):
            s[g] = s2[g]
            for i in range(H):
                v = hv2[g, i]
                hv[g, i] = v if v > 0 else 0.0
    # logits on the grid and the wrong set per interval
    L = np.empty((G, O))
    for g in range(G):
        for o in range(O):
            acc = 0.0
            for k in range(H):
                acc += Wo[o, k] * hv[g, k]
            L[g, o] = acc
    pieces = np.empty((maxp, 2)); npc = 0
    roots = np.empty((maxp, 4)); nr = 0
    for g in range(G - 1):
        a = s[g]; b = s[g + 1]
        if b <= a:
            continue
        k = kneg if b <= 0.0 else kpos
        side = 0 if b <= 0.0 else 1
        clo = a; chi = b; jlo = -1; jhi = -1; empty = False
        sc = 0.0
        for o in range(O):
            sc = max(sc, abs(L[g, o]), abs(L[g + 1, o]))
        for j in range(O):
            if j == k:
                continue
            f0 = L[g, k] - L[g, j]; f1 = L[g + 1, k] - L[g + 1, j]
            if j < k and abs(f0) <= 1e-12 * (1 + sc) and abs(f1) <= 1e-12 * (1 + sc):
                empty = True; break               # exact tie, first index wins (numpy argmax rule)
            if f0 >= 0 and f1 >= 0:
                continue
            if f0 < 0 and f1 < 0:
                empty = True; break
            r = a + (b - a) * (f0 / (f0 - f1))
            if f0 < 0:
                if r > clo:
                    clo = r; jlo = j
            else:
                if r < chi:
                    chi = r; jhi = j
        if empty or chi <= clo:
            if npc < maxp:
                pieces[npc, 0] = a; pieces[npc, 1] = b; npc += 1
            continue
        if clo > a:
            if npc < maxp:
                pieces[npc, 0] = a; pieces[npc, 1] = clo; npc += 1
            if nr < maxp:
                roots[nr, 0] = clo; roots[nr, 1] = k; roots[nr, 2] = jlo; roots[nr, 3] = side; nr += 1
        if chi < b:
            if npc < maxp:
                pieces[npc, 0] = chi; pieces[npc, 1] = b; npc += 1
            if nr < maxp:
                roots[nr, 0] = chi; roots[nr, 1] = k; roots[nr, 2] = jhi; roots[nr, 3] = side; nr += 1
    st = 0 if (npc < maxp and nr < maxp) else 2
    return pieces[:npc], roots[:nr], gates[:ng], st


@njit(cache=True)
def signature(Wi, Wh, Wo, w, d, tw, pieces, roots, lab):
    """Combinatorial type of a line: ordered decision roots (side, k, j, gate pattern of the region at the root), which
    model-free boundaries (window ends, s = 0) are wrong, and the model-free crease label."""
    hsh = np.uint64(14695981039346656037) ^ np.uint64(lab + 7)
    P = np.uint64(1099511628211)
    for r in range(roots.shape[0]):
        x = w + roots[r, 0] * d
        pat, J, G = _forward_pattern(Wi, Wh, x)
        for q in range(3):
            hsh = (hsh ^ np.uint64(int(roots[r, 1 + q]) + 11)) * P
        hsh = (hsh ^ _pat_hash(pat)) * P
    lo_wrong = 0; hi_wrong = 0; zl = 0; zr = 0
    for p in range(pieces.shape[0]):
        if pieces[p, 0] == -tw:
            lo_wrong = 1
        if pieces[p, 1] == tw:
            hi_wrong = 1
        if pieces[p, 1] == 0.0:
            zl = 1
        if pieces[p, 0] == 0.0:
            zr = 1
    hsh = (hsh ^ np.uint64(lo_wrong + 2 * hi_wrong + 4 * zl + 8 * zr + 100)) * P
    return hsh


def run(model, j, U, want_gates=False, want_sig=False):
    """Per-line value g (stratum weight x wrong Student-t mass) and, on request, decision roots / gate roots /
    signatures, for stratum j and nodes U."""
    Wi, Wh, Wo = model
    w, d, tw, kneg, kpos, wt, lab = lines(j, U)
    n = len(U)
    mass = np.zeros(n); sig = np.zeros(n, np.uint64); R = []; Q = []
    for i in range(n):
        pc, rt, gt, st = walk_line(Wi, Wh, Wo, w[i], d[i], tw[i], kneg[i], kpos[i], want_gates)
        assert st == 0, (j, i, st)
        mass[i] = (kcdf(pc[:, 1]) - kcdf(pc[:, 0])).sum()
        if want_sig:
            sig[i] = signature(Wi, Wh, Wo, w[i], d[i], tw[i], pc, rt, lab[i])
        if want_gates:
            R.append(rt); Q.append(gt)
    out = {"g": wt * mass, "mass": mass, "w": w, "d": d, "tw": tw}
    if want_sig:
        out["sig"] = sig
    if want_gates:
        out["roots"] = R; out["gates"] = Q
    return out


def sensitivity(model, w, d, roots, gates):
    """rho = |P c| / |c.d| for decision roots (c = grad_x (L_k - L_j)) and gate roots (c = grad_x z_{t,i}),
    plus the Student-t density weight k(s*) at decision roots."""
    Wi, Wh, Wo = model
    nu = T - 1
    from scipy.stats import t as tdist
    rE, kE, rG = [], [], []
    for r in roots:
        x = w + r[0] * d
        pat, J, G = _forward_pattern(Wi, Wh, x)
        c = (Wo[int(r[1])] - Wo[int(r[2])]) @ J
        cd = c @ d
        rE.append(np.linalg.norm(c - cd * d) / abs(cd))
        kE.append(tdist.pdf(r[0] * np.sqrt(nu), nu) * np.sqrt(nu))
    for q in gates:
        x = w + q[0] * d
        pat, J, G = _forward_pattern(Wi, Wh, x)
        c = G[int(q[1]), int(q[2])]
        cd = c @ d
        rG.append(np.linalg.norm(c - cd * d) / abs(cd))
    return np.array(rE), np.array(kE), np.array(rG)

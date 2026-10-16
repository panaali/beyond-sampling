"""Two boundary layers + remainder, all exact-in-the-normal-direction except the remainder indicator.

error = E[fA(x)] + E[fB(x)] + E[R(x)]
  A: line through x along (e_r2 - e_r3)/sqrt2, rank-preserving segment; band = connected component of the
     tie point t=0 where the output is neither the target (rank 2) nor the current rank-1 index.
  B: line through x along (e_r1 - e_r2)/sqrt2, segment where r1,r2 stay the top two; band = component of
     t=0 where the output is the current max ("picks the max" error at the 1st-vs-2nd tie).
  fA, fB = Gaussian mass of the band / mass of the segment  (exact piecewise-linear integration)
  R(x) = 1[x wrong, x not in A-band, x not in B-band]       (one indicator; free if x lies in a walked piece)
A-band and B-band are disjoint (different outputs), each is a function of its line only, so each term is an
exact conditional expectation of a disjoint piece of the error indicator: the sum is unbiased.
"""
import numpy as np
from scipy.special import ndtr
from local import _pass

T = 10
SQ2 = np.sqrt(2.0)


def _env(lg, dlg, a0, end, tgt, excl, badk):
    # tgt: correct output on this side; component walked = where output != tgt
    N = len(lg); rows = np.arange(N)
    a = a0.copy(); wrong = np.zeros(N); found = np.zeros(N, bool); live = a < end
    for _ in range(60):
        live &= ~found & (a < end)
        if not live.any():
            break
        val = lg + dlg * (a - a0)[:, None]
        k = np.argmax(val + dlg * 1e-7, 1)
        vk, dk = val[rows, k], dlg[rows, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = a[:, None] + (vk[:, None] - val) / (dlg - dk[:, None])
        c = np.where((dlg > dk[:, None]) & (c > a[:, None] + 1e-13), c, np.inf).min(1)
        b = np.minimum(end, c)
        wr = k != tgt                                   # still inside the wrong component
        bad = np.where(badk >= 0, k == badk, k != excl)  # output of this band's type
        found |= live & ~wr
        w = live & wr
        wrong += np.where(w & bad, ndtr(b) - ndtr(a), 0.0)
        a = np.where(w, b, a)
    return wrong, a, found


def _side(model, x0, d, lg, dlg, brk, end, tgt, excl, badk, sign, max_pieces=60):
    N = len(x0)
    wrong, stop, found = _env(lg, dlg, np.zeros(N), np.minimum(brk, end), tgt, excl, badk)
    passes = np.zeros(N, int)
    cont = ~found & (stop < end)
    for _ in range(max_pieces):
        if not cont.any():
            break
        i = np.nonzero(cont)[0]
        S = stop[i] + 1e-9
        lg2, dlg2, up, _ = _pass(model, x0[i], sign * d[i], S)
        passes[i] += 1
        lg2 = lg2 - dlg2 * 1e-9
        w2, st2, f2 = _env(lg2, dlg2, stop[i], np.minimum(up, end[i]), tgt[i], excl[i], badk[i])
        wrong[i] += w2; stop[i] = st2; found[i] = f2
        cont[i] = ~f2 & (st2 < end[i])
    return wrong, stop, passes


def band(model, x0, d, Tp, Tn, side_pos, side_neg):
    """side_* = (tgt, excl, badk) arrays for t>0 and t<0. Returns band mass, right stop, left stop, passes,
    and the first piece info (lg, dlg, up, dn) at the tie."""
    N = len(x0)
    lg, dlg, up, dn = _pass(model, x0, d, np.zeros(N))
    wr, sr, pr = _side(model, x0, d, lg, dlg, up, Tp, *side_pos, +1)
    wl, sl, pl = _side(model, x0, d, lg, -dlg, -dn, Tn, *side_neg, -1)
    return wr + wl, sr, sl, 1 + pr + pl, (lg, dlg, up, dn)


def line_A(x):
    N = len(x); o = np.argsort(-x, 1); rows = np.arange(N)
    r1, a, b, r4 = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    d = np.zeros_like(x); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    t0 = (x[rows, a] - x[rows, b]) / SQ2
    yb = (x[rows, a] + x[rows, b]) / 2
    Tm = SQ2 * np.minimum(x[rows, r1] - yb, yb - x[rows, r4])
    neg1 = -np.ones(N, int)
    return dict(x0=x - t0[:, None] * d, d=d, t0=t0, Tp=Tm, Tn=Tm, ybar=yb,
                pos=(a, r1, neg1), neg=(b, r1, neg1))


def line_B(x):
    N = len(x); o = np.argsort(-x, 1); rows = np.arange(N)
    a, b, r3 = o[:, 0], o[:, 1], o[:, 2]
    d = np.zeros_like(x); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    t0 = (x[rows, a] - x[rows, b]) / SQ2
    yb = (x[rows, a] + x[rows, b]) / 2
    Tm = SQ2 * (yb - x[rows, r3])
    z = np.zeros(N, int)
    return dict(x0=x - t0[:, None] * d, d=d, t0=t0, Tp=Tm, Tn=Tm, ybar=yb,
                pos=(b, z, a), neg=(a, z, b))


def frac(model, L):
    m, sr, sl, p, first = band(model, L["x0"], L["d"], L["Tp"], L["Tn"], L["pos"], L["neg"])
    tot = ndtr(L["Tp"]) - ndtr(-L["Tn"])
    inb = (L["t0"] <= sr) & (L["t0"] >= -sl)
    return m / tot, inb, p, first


def known_output(L, first):
    """Output at x if x lies in the first linear piece of the line (else -1)."""
    lg, dlg, up, dn = first
    t0 = L["t0"]
    ok = (t0 < up) & (t0 > dn)
    out = np.argmax(lg + dlg * t0[:, None], 1)
    return np.where(ok, out, -1)


def point_out(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def per_sample(model, x, do_A=True, do_B=True, do_R=True):
    """Returns dict of per-sample terms and passes."""
    out = {}
    passes = np.zeros(len(x), int)
    LA, LB = line_A(x), line_B(x)
    fA, inA, pA, firstA = frac(model, LA); passes += pA
    fB, inB, pB, firstB = frac(model, LB); passes += pB
    ko = known_output(LA, firstA)
    ko = np.where(ko >= 0, ko, known_output(LB, firstB))
    need = ko < 0
    if need.any():
        ko[need] = point_out(model, x[need]); passes[need] += 1
    tgt = np.argsort(x, 1)[:, -2]; r1 = np.argmax(x, 1)
    inA = inA & (ko != r1); inB = inB & (ko == r1)
    R = (ko != tgt) & ~inA & ~inB
    return dict(fA=fA, fB=fB, R=R.astype(float), passes=passes, ybarA=LA["ybar"], TA=LA["Tp"],
                ybarB=LB["ybar"], TB=LB["Tn"])

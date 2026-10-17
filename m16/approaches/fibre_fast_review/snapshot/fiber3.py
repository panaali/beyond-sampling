"""T73: "fibre-wedge" accuracy estimate for bias-free 2nd-argmax RNNs with T = 3 (any H). Deductive, closed form.

Geometry (model-independent, exact partition of the sphere):
  * The true class changes only on the circles x_i = x_j. Each circle is split by x_k = x_i = x_j into two
    semicircles (arcs), "k above" (s = +1) and "k below" (s = -1); both run from c = (1,1,1)/sqrt3 to -c:
    p(theta) = cos(theta) c + sin(theta) s d_k, theta in [0, pi], d_k = unit(e_k - (e_i + e_j)/2).
  * Every point x off the circles is written x ~ p(theta) + u n' (n' = +-n, n = unit(e_i - e_j), u = tan(phi) >= 0,
    phi = angular distance from the circle) for the true boundary nearest to it among the two boundaries of its
    ordering chamber; the fibre stops at the equidistance plane (clamp). Area element cos(phi) dphi dtheta, so a
    fibre segment u in (u_lo, u_hi) contributes [S(u_hi) - S(u_lo)] dtheta / (4 pi), S(u) = u / sqrt(1 + u^2).
Mechanism (all from the weights):
  * Along each arc the gate pattern is traced EXACTLY (`trace_arc`): step-t gate rows A_t (A_1 = Wi e0^T,
    A_{t+1} = Wh diag(g_t) A_t + Wi e_t^T) switch where A_t,k . p(theta) = 0 (closed-form roots). This gives arc
    segments of constant pattern with linear logits L = Wo diag(g_T) A_T.
  * Tier F0 (frozen): across the thin wedge the pattern is frozen at its arc value. On the fibre the logits are
    a + u b (a = L p, b = L n'), the network's c-vs-d boundary is at u_d(theta) = -(w.p)/(w.n'), w = L_c - L_d, a
    trig-linear function alpha cos(theta) + beta sin(theta). The wrong set on each fibre is read from the envelope
    of T lines; its endpoints are u_d's, 0, or the clamp, so its mass is a sum of closed forms
    int S(R cos(theta - theta0)) dtheta = asin(R sin(theta - theta0) / sqrt(1 + R^2)). The arc segment is split at
    the zeros and mutual crossings of these functions so the envelope structure is constant on each piece.
accuracy = 1 - sum of wrong masses.
"""
import math

import numpy as np

T3 = 3
_C = np.ones(3) / math.sqrt(3)


class Ops:
    def __init__(self):
        self.n = 0

    def add(self, k):
        self.n += k


# ----------------------------------------------------------------------------- model-independent fibre families
def families():
    """List of arcs: dict(P, U, n, sides=[(nn, truth, clamp_normal)]) for T = 3. Computed once (not charged)."""
    E = np.eye(3); out = []
    for (i, j, k) in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        n = (E[i] - E[j]) / math.sqrt(2)
        d = E[k] - 0.5 * (E[i] + E[j]); d /= np.linalg.norm(d)
        for s in (1, -1):
            sides = []
            for sg, hi, lo in ((1, i, j), (-1, j, i)):          # fibre direction towards x_hi > x_lo
                if s == 1:      # x_k > x_hi > x_lo: 2nd = hi; clamp x_hi - x_lo < x_k - x_hi
                    truth = hi; w = E[k] - 2 * E[hi] + E[lo]
                else:           # x_hi > x_lo > x_k: 2nd = lo; clamp x_hi - x_lo < x_lo - x_k
                    truth = lo; w = 2 * E[lo] - E[hi] - E[k]
                sides.append((sg * n, truth, w))
            out.append(dict(P=_C.copy(), U=s * d, n=n, sides=sides, circle=(i, j, k), s=s))
    return out


FAMS = families()


# ----------------------------------------------------------------------------- exact trace of the pattern along an arc
def _root(a, b, lo, hi):
    """theta in (lo, hi) with a cos + b sin = 0 (one root per pi)."""
    t = math.atan2(a, -b) % math.pi
    return t if lo < t < hi else None


def trace_arc(model, P, U, lo, hi, ops):
    """Segments (t0, t1, L) of the arc p(t) = cos t P + sin t U, t in (lo, hi), on which the gate pattern is
    constant; L = linear logit map (T x T) there. All gate switches on the arc are found exactly."""
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]
    segs = [(lo, hi, np.outer(Wi, np.eye(T)[0]))]           # step-1 rows, pattern not yet applied
    for t in range(T):
        nxt = []
        for (a0, a1, A) in segs:
            pa = A @ P; pb = A @ U; ops.add(2 * H * (t + 1))
            cuts = sorted(r for r in (_root(pa[m], pb[m], a0, a1) for m in range(H)) if r is not None)
            ops.add(3 * H)
            bs = [a0] + cuts + [a1]
            for q in range(len(bs) - 1):
                tm = 0.5 * (bs[q] + bs[q + 1])
                g = (pa * math.cos(tm) + pb * math.sin(tm)) > 0; ops.add(2 * H + 2)
                Ag = A * g[:, None]
                if t + 1 < T:
                    An = Wh @ Ag; An[:, t + 1] += Wi; ops.add(H * int(g.sum()) * (t + 1))
                    nxt.append((bs[q], bs[q + 1], An))
                else:
                    L = Wo @ Ag; ops.add(T * int(g.sum()) * T)
                    nxt.append((bs[q], bs[q + 1], L, bool(g.any())))
        segs = nxt
    return segs


# ----------------------------------------------------------------------------- closed-form wedge masses
def _Sint(al, be, t0, t1):
    """int_{t0}^{t1} S(al cos t + be sin t) dt, S(u) = u / sqrt(1 + u^2)."""
    R = math.hypot(al, be)
    if R == 0.0:
        return 0.0
    th = math.atan2(be, al); k = R / math.sqrt(1 + R * R)
    return math.asin(max(-1.0, min(1.0, k * math.sin(t1 - th)))) - math.asin(max(-1.0, min(1.0, k * math.sin(t0 - th))))


def _wrong_intervals(a, b, truth, ucl, live):
    """Wrong set of u in (0, ucl) for logits a + u b (argmax; dead -> class 0). Returns [(lo_id, hi_id)] where an id
    is 'z' (u = 0), 'c' (clamp) or a class d (crossing of truth with d)."""
    T = len(a)
    if not live:
        return [] if truth == 0 else [("z", "c")]
    # breakpoints u where truth ties with some d, inside (0, ucl)
    pts = []
    for d in range(T):
        if d == truth or b[d] == b[truth]:
            continue
        u = (a[truth] - a[d]) / (b[d] - b[truth])
        if 0 < u < ucl:
            pts.append((u, d))
    pts.sort()
    edges = [(0.0, "z")] + pts + [(ucl, "c")]
    out = []
    for q in range(len(edges) - 1):
        um = 0.5 * (edges[q][0] + edges[q + 1][0])
        l = a + um * b
        if int(np.argmax(l)) != truth:
            if out and out[-1][1] == edges[q][1]:
                out[-1] = (out[-1][0], edges[q + 1][1])
            else:
                out.append((edges[q][1], edges[q + 1][1]))
    return out


def seg_wrong_mass(L, live, P, U, nn, truth, wcl, t0, t1, ops):
    """Frozen-pattern wrong mass (x 4pi) over arc segment (t0, t1) on fibre side nn."""
    T = L.shape[0]
    # trig-linear coefficient functions u_f(t) = al_f cos t + be_f sin t for f in {class d crossings, clamp}
    fun = {}
    Ln = L @ nn; LP = L @ P; LU = L @ U; ops.add(3 * 3 * T)
    for d in range(T):
        if d == truth:
            continue
        den = Ln[d] - Ln[truth]
        if abs(den) < 1e-300:
            continue
        # tie: LP_c - LP_d + u (Ln_c - Ln_d) = 0 -> u = (a_c - a_d) / (b_d - b_c)
        fun[d] = ((LP[truth] - LP[d]) / den, (LU[truth] - LU[d]) / den); ops.add(4)
    wn = -(wcl @ nn)
    fun["c"] = ((wcl @ P) / wn, (wcl @ U) / wn); ops.add(9 + 2)
    # split points: zeros of each function and mutual crossings
    keys = list(fun)
    cuts = set()
    for x in range(len(keys)):
        al, be = fun[keys[x]]
        r = _root(al, be, t0, t1); ops.add(4)
        if r is not None:
            cuts.add(r)
        for y in range(x + 1, len(keys)):
            a2, b2 = fun[keys[y]]
            r = _root(al - a2, be - b2, t0, t1); ops.add(6)
            if r is not None:
                cuts.add(r)
    bs = [t0] + sorted(cuts) + [t1]
    tot = 0.0
    for q in range(len(bs) - 1):
        tm = 0.5 * (bs[q] + bs[q + 1]); cm, sm = math.cos(tm), math.sin(tm)
        p = cm * P + sm * U
        a = LP * cm + LU * sm; ucl = fun["c"][0] * cm + fun["c"][1] * sm; ops.add(2 * T + 4)
        if ucl <= 0:
            continue
        iv = _wrong_intervals(a, Ln, truth, ucl, live); ops.add(6 * T)
        for lo, hi in iv:
            for sgn, idf in ((1.0, hi), (-1.0, lo)):
                if idf == "z":
                    continue
                al, be = fun[idf]
                tot += sgn * _Sint(al, be, bs[q], bs[q + 1]); ops.add(12)
    return tot


def est_fiber3(model, ops=None, info=None):
    ops = ops if ops is not None else Ops()
    tot = 0.0; by = {}
    for f in FAMS:
        segs = trace_arc(model, f["P"], f["U"], 0.0, math.pi, ops)
        e = 0.0
        for (t0, t1, L, live) in segs:
            for nn, truth, wcl in f["sides"]:
                e += seg_wrong_mass(L, live, f["P"], f["U"], nn, truth, wcl, t0, t1, ops)
        tot += e
        if info is not None:
            by[(f["circle"], f["s"])] = e / (4 * math.pi)
            info.setdefault("segments", 0)
            info["segments"] += len(segs)
    if info is not None:
        info["err_by"] = by
    return 1.0 - tot / (4 * math.pi)

"""T113: independent re-implementation of RECIPE112 (rule "margin_all"), written for this review in plain Python
scalars, generic H, T = 3. It runs on floats, or on the `Num` type below, which counts every executed arithmetic
operation, abs, comparison, sqrt and atan2 on model-dependent numbers (automatic op count; no hand-assigned charges).

Recipe (as described in understand_m43/README.md, T112 section):
  * 24 half-chambers (ordering chamber x chart sign(x0), split by x_max + x_min = 2 x_med); built here.
  * Step 2: freeze every cutting step-2 gate at its state at the arc reference point (mean of the cell's vertices on
    the true boundary plane, else mean of all vertices). Gate k triggers if on its minority side the bound
    |a_k.v| w_k, w_k = max_o sum_j |Wo[t,j]-Wo[o,j]| |Wh[j,k]|, reaches lip * |m(v)| (m = margin of the frozen network,
    full forward pass) at a test point (minority vertices and gate/edge crossing points), or m changes sign over the
    test points with max shift >= eps * max|m|. All triggering gates split the cell together; children re-tested
    (already-split gates excluded).
  * Step 3: same with the linear frozen readout and w_k = max_o |Wo[t,k]-Wo[o,k]|.
  * Final cells: error mass = parts cut off by the halfspaces l_truth >= l_o (all-off cell: tie -> class 0).
Knobs for the sensitivity audit (frozen values in brackets): eps [0.1], lip [1.0], tie [1e-9], gtol [1e-12] (gate
"does not cut" tolerance, relative), ctol [1e-12] (clip snap tolerance, relative), atol [1e-9] (arc vertex test).
"""
import itertools
import math

# --------------------------------------------------------------------------------------------- op counting type
C = {"n": 0}


def val(x):
    return x.x if isinstance(x, Num) else x


class Num:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = float(x)

    def _b(self, o, f):
        C["n"] += 1; return Num(f(self.x, val(o)))

    def __add__(self, o): return self._b(o, lambda a, b: a + b)
    def __radd__(self, o): return self._b(o, lambda a, b: b + a)
    def __sub__(self, o): return self._b(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._b(o, lambda a, b: b - a)
    def __mul__(self, o): return self._b(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._b(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._b(o, lambda a, b: a / b)
    def __rtruediv__(self, o): return self._b(o, lambda a, b: b / a)
    def __neg__(self): return Num(-self.x)
    def __abs__(self): C["n"] += 1; return Num(abs(self.x))
    def __lt__(self, o): C["n"] += 1; return self.x < val(o)
    def __le__(self, o): C["n"] += 1; return self.x <= val(o)
    def __gt__(self, o): C["n"] += 1; return self.x > val(o)
    def __ge__(self, o): C["n"] += 1; return self.x >= val(o)
    def __float__(self): return self.x


def _sqrt(a):
    if isinstance(a, Num):
        C["n"] += 1; return Num(math.sqrt(a.x))
    return math.sqrt(a)


def _atan2(a, b):
    if isinstance(a, Num) or isinstance(b, Num):
        C["n"] += 1; return Num(math.atan2(val(a), val(b)))
    return math.atan2(a, b)


# ------------------------------------------------------------------------------------------------- geometry
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def unit(p):
    n = _sqrt(dot(p, p)); return (p[0] / n, p[1] / n, p[2] / n)


def clip(poly, n, ctol):
    s = [dot(n, v) for v in poly]
    if ctol > 0:
        big = max(abs(x) for x in s); t = big * ctol
        s = [0.0 if abs(x) <= t else x for x in s]
    if min(s) >= 0:
        return poly, []
    if max(s) <= 0:
        return [], poly
    a, b = [], []
    m = len(poly)
    for i in range(m):
        si, sj = s[i], s[(i + 1) % m]
        if si >= 0:
            a.append(poly[i])
        if si <= 0:
            b.append(poly[i])
        if (si > 0 and sj < 0) or (si < 0 and sj > 0):
            u, w = abs(si), abs(sj); vi, vj = poly[i], poly[(i + 1) % m]
            p = unit((u * vj[0] + w * vi[0], u * vj[1] + w * vi[1], u * vj[2] + w * vi[2]))
            a.append(p); b.append(p)
    return (a if len(a) >= 3 else []), (b if len(b) >= 3 else [])


def area(poly):
    if len(poly) < 3:
        return 0.0
    a = poly[0]; tot = 0.0
    for i in range(1, len(poly) - 1):
        b, c = poly[i], poly[i + 1]
        tot = tot + 2.0 * _atan2(abs(dot(a, cross(b, c))), 1.0 + dot(a, b) + dot(a, c) + dot(b, c))
    return tot / (4 * math.pi)


def _e(i):
    v = [0.0, 0.0, 0.0]; v[i] = 1.0; return tuple(v)


def _halves():
    import numpy as np
    out = []
    for s0 in (1.0, -1.0):
        for mx, md, mn in itertools.permutations(range(3)):
            N = np.array([np.subtract(_e(mx), _e(md)), np.subtract(_e(md), _e(mn)), np.multiply(s0, _e(0))])
            R = np.linalg.inv(N)
            poly = [tuple(float(x) for x in R[:, i] / np.linalg.norm(R[:, i])) for i in range(3)]
            split = tuple(_e(mx)[i] + _e(mn)[i] - 2 * _e(md)[i] for i in range(3))
            for near in (mx, mn):
                nn = tuple(-x for x in split) if near == mx else split
                p, _ = clip(poly, nn, 0.0)
                nt = tuple(np.subtract(_e(mx), _e(md))) if near == mx else tuple(np.subtract(_e(md), _e(mn)))
                out.append(dict(s0=s0, truth=md, n_true=tuple(float(x) for x in nt), poly=p))
    return out


HALVES = _halves()


# ------------------------------------------------------------------------------------------------- recipe
class P:
    eps = 0.1; lip = 1.0; tie = 1e-9; gtol = 1e-12; ctol = 1e-12; atol = 1e-9
    # diagnostic variants (NOT part of RECIPE112; off by default):
    dense = 0        # > 0: also test a barycentric grid (this many steps per fan triangle) inside the minority region
    joint = False    # also test the SUM of the shift bounds of all frozen cutting gates against the margin


def _gates(poly, rows, ref, done, gtol):
    S = [[dot(a, v) for v in poly] for a in rows]
    g = []; cut = []
    for k, s in enumerate(S):
        big = max(abs(x) for x in s); t = big * gtol
        if min(s) >= -t:
            g.append(1.0 if max(s) > t else (1.0 if dot(rows[k], ref) > 0 else 0.0))
        elif max(s) <= t:
            g.append(0.0)
        else:
            g.append(1.0 if dot(rows[k], ref) > 0 else 0.0)
            if k not in done:
                cut.append(k)
    return S, g, cut


def _ref(poly, n_true, atol):
    on = [v for v in poly if abs(dot(n_true, v)) < atol]
    base = on if len(on) >= 2 else poly
    return tuple(sum(v[i] for v in base) / len(base) for i in range(3))


def _grid(poly, N):
    out = []
    a0 = poly[0]
    for i in range(1, len(poly) - 1):
        b, c = poly[i], poly[i + 1]
        for p in range(N + 1):
            for q in range(N + 1 - p):
                r = N - p - q
                if 0 in (p, q, r) and N > 1:
                    continue
                out.append(unit(tuple((p * a0[j] + q * b[j] + r * c[j]) / N for j in range(3))))
    return out


def _minority(poly, a, gk):
    ins, outp = clip(poly, a, 0.0)
    return ins if gk == 0 else outp


def _test(poly, s, gk, wk, margin, pr, a=None):
    """Does freezing this gate (frozen state gk, pre-activations s at the vertices) possibly matter?"""
    m = len(poly); pts = []; sh = []
    if pr.dense and a is not None:
        mp = _minority(poly, a, gk)
        if mp:
            for v in _grid(mp, pr.dense):
                pts.append(v); sh.append(abs(dot(a, v)) * wk)
    for q in range(m):
        if (s[q] > 0) != (gk > 0):
            pts.append(poly[q]); sh.append(abs(s[q]) * wk)
    for q in range(m):
        r = (q + 1) % m
        if (s[q] > 0) != (s[r] > 0) and s[q] != 0 and s[r] != 0:
            u, w = abs(s[q]), abs(s[r]); vq, vr = poly[q], poly[r]
            pts.append(unit((u * vr[0] + w * vq[0], u * vr[1] + w * vq[1], u * vr[2] + w * vq[2]))); sh.append(0.0)
    mm = margin(pts)
    score = max(x / (abs(y) + 1e-300) for x, y in zip(sh, mm))
    flips = min(mm) < 0 < max(mm) and max(sh) >= pr.eps * max(abs(y) for y in mm)
    return score >= pr.lip * (1 - pr.tie) or flips


def _joint(poly, rows, g, cut, w, margin, pr):
    """Diagnostic: at the cell's vertices, all crossing points (and grid points if dense), sum the shift bounds of
    every frozen cutting gate whose minority side contains the point; trigger those gates if the sum reaches the
    margin somewhere."""
    pts = list(poly)
    for k in cut:
        s = [dot(rows[k], v) for v in poly]; m = len(poly)
        for q in range(m):
            r = (q + 1) % m
            if (s[q] > 0) != (s[r] > 0) and s[q] != 0 and s[r] != 0:
                u, ww = abs(s[q]), abs(s[r]); vq, vr = poly[q], poly[r]
                pts.append(unit((u * vr[0] + ww * vq[0], u * vr[1] + ww * vq[1], u * vr[2] + ww * vq[2])))
        if pr.dense:
            mp = _minority(poly, rows[k], g[k])
            if mp:
                pts += _grid(mp, pr.dense)
    mm = margin(pts)
    for v, y in zip(pts, mm):
        ks = [k for k in cut if (dot(rows[k], v) > 0) != (g[k] > 0)]
        tot = sum(abs(dot(rows[k], v)) * w[k] for k in ks)
        if ks and tot >= pr.lip * abs(y) * (1 - pr.tie):
            return ks
    return []


def _split_all(poly, rows, trig, ctol):
    pieces = [poly]
    for k in trig:
        nxt = []
        for q in pieces:
            a, b = clip(q, rows[k], ctol)
            nxt += [x for x in (a, b) if x]
        pieces = nxt
    return pieces


def _step2(poly, A2, model, t, n_true, pr, st, done=()):
    Wi, Wh, Wo = model; H = len(Wi)
    ref = _ref(poly, n_true, pr.atol)
    S, g, cut = _gates(poly, A2, ref, done, pr.gtol)
    if not done:
        st["cut"] += len(cut)
    if not cut:
        return [(poly, g)]
    w = [max(sum(abs(Wo[t][j] - Wo[o][j]) * abs(Wh[j][k]) for j in range(H)) for o in range(3) if o != t)
         for k in range(H)]
    A3f = [tuple(sum(Wh[i][j] * A2[j][c] for j in range(H) if g[j]) + (Wi[i] if c == 2 else 0.0) for c in range(3))
           for i in range(H)]

    def margin(pts):
        out = []
        for v in pts:
            z = [dot(A3f[i], v) for i in range(H)]
            z = [x if x > 0 else 0.0 for x in z]
            if not any(val(x) > 0 for x in z):
                out.append(0.0); continue
            l = [sum(Wo[o][i] * z[i] for i in range(H) if val(z[i]) > 0) for o in range(3)]
            out.append(l[t] - max(l[o] for o in range(3) if o != t))
        return out

    trig = [k for k in cut if _test(poly, S[k], g[k], w[k], margin, pr, A2[k])]
    if not trig and pr.joint:
        trig = _joint(poly, A2, g, cut, w, margin, pr)
    if not trig:
        return [(poly, g)]
    st["resolved"] += len(trig)
    out = []
    for q in _split_all(poly, A2, trig, pr.ctol):
        out += _step2(q, A2, model, t, n_true, pr, st, tuple(done) + tuple(trig))
    return out


def _step3(poly, A3, Wo, t, n_true, pr, st, done=()):
    H = len(A3)
    ref = _ref(poly, n_true, pr.atol)
    S, g, cut = _gates(poly, A3, ref, done, pr.gtol)
    if not done:
        st["cut"] += len(cut)
    if not cut:
        return [(poly, g)]
    w = [max(abs(Wo[t][k] - Wo[o][k]) for o in range(3) if o != t) for k in range(H)]
    L = [tuple(sum(Wo[o][k] * A3[k][c] for k in range(H) if g[k]) for c in range(3)) for o in range(3)]
    D = [tuple(L[t][c] - L[o][c] for c in range(3)) for o in range(3)]
    allzero = not any(g)

    def margin(pts):
        if allzero:
            return [0.0] * len(pts)
        return [min(dot(D[o], v) for o in range(3) if o != t) for v in pts]

    trig = [k for k in cut if _test(poly, S[k], g[k], w[k], margin, pr, A3[k])]
    if not trig and pr.joint:
        trig = _joint(poly, A3, g, cut, w, margin, pr)
    if not trig:
        return [(poly, g)]
    st["resolved"] += len(trig)
    out = []
    for q in _split_all(poly, A3, trig, pr.ctol):
        out += _step3(q, A3, Wo, t, n_true, pr, st, tuple(done) + tuple(trig))
    return out


def rec112(model, pr=P, st=None):
    """model = (Wi, Wh, Wo) as nested lists of floats (or Num). Returns the accuracy estimate."""
    Wi, Wh, Wo = model; H = len(Wi)
    st = st if st is not None else {}
    for k in ("cut", "resolved", "cells"):
        st.setdefault(k, 0)
    err = 0.0
    for h in HALVES:
        s0, t, n_true = h["s0"], h["truth"], h["n_true"]
        g1 = [1.0 if s0 * Wi[j] > 0 else 0.0 for j in range(H)]
        A2 = [(sum(Wh[k][j] * Wi[j] for j in range(H) if g1[j]), Wi[k], 0.0) for k in range(H)]
        for p2, g2 in _step2(h["poly"], A2, model, t, n_true, pr, st):
            A3 = [tuple(sum(Wh[k][j] * A2[j][c] for j in range(H) if g2[j]) + (Wi[k] if c == 2 else 0.0)
                        for c in range(3)) for k in range(H)]
            for p3, g3 in _step3(p2, A3, Wo, t, n_true, pr, st):
                st["cells"] += 1
                if not any(g3):
                    if t != 0:
                        err = err + area(p3)
                    continue
                L = [tuple(sum(Wo[o][k] * A3[k][c] for k in range(H) if g3[k]) for c in range(3)) for o in range(3)]
                q = p3
                for o in range(3):
                    if o == t:
                        continue
                    a, b = clip(q, tuple(L[t][c] - L[o][c] for c in range(3)), pr.ctol)
                    if b:
                        err = err + area(b)
                    q = a
                    if not q:
                        break
    return 1.0 - val(err)


def as_lists(model, wrap=False):
    f = Num if wrap else float
    Wi, Wh, Wo = model
    return [f(x) for x in Wi], [[f(x) for x in r] for r in Wh], [[f(x) for x in r] for r in Wo]


def counted(model, pr=P):
    """(estimate, automatic op count) with every model-dependent scalar operation counted."""
    C["n"] = 0
    r = rec112(as_lists(model, True), pr)
    return r, C["n"]


def sample_ops(H):
    """Same counter on one sampling step: 3 RNN steps + ReLU, readout, argmax, 2nd-argmax label, compare."""
    import random
    C["n"] = 0
    rnd = random.Random(0)
    Wi = [Num(rnd.gauss(0, 1)) for _ in range(H)]; Wh = [[Num(rnd.gauss(0, 1)) for _ in range(H)] for _ in range(H)]
    Wo = [[Num(rnd.gauss(0, 1)) for _ in range(H)] for _ in range(3)]
    x = [Num(rnd.gauss(0, 1)) for _ in range(3)]
    h = [x[0] * Wi[k] for k in range(H)]; h = [a if a > 0 else 0.0 for a in h]
    for t in (1, 2):
        pre = [x[t] * Wi[k] + sum((Wh[k][j] * h[j] for j in range(1, H)), Wh[k][0] * h[0]) for k in range(H)]
        h = [a if a > 0 else 0.0 for a in pre]
    lo = [sum((Wo[o][j] * h[j] for j in range(1, H)), Wo[o][0] * h[0]) for o in range(3)]
    am = 0
    for o in (1, 2):
        if lo[o] > lo[am]:
            am = o
    _ = (x[0] > x[1], x[1] > x[2], x[0] > x[2])
    return C["n"]

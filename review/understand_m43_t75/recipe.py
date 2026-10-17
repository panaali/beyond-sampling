"""T84 review: independent re-implementation of the T75 'resolve-by-impact' recipe, written from the README
description of est75 (not copied). Pure Python scalars, so it runs with plain floats (fast enough) or with the
`Num` counting type of count.py (every executed + - * / abs compare sqrt atan2 is counted automatically).

Own model-independent geometry: the 24 half-chambers are built here from the definition (ordering chamber x chart
sign(x0), split by the plane x_max + x_min = 2 x_med into the half next to x_med = x_max and the half next to
x_med = x_min); their solid angles must sum to 1.

Variants (for the selection-risk audit):
  eps     relative threshold (frozen T75: 0.1); a pair (eps_step2, eps_step3) sets the steps separately
  wmode   downstream weight of a gate: 'max' (T75), 'one' (depth only), 'l2' (Euclidean norm of downstream column)
  smode   scale of the surely-on neurons: 'max' (T75: sum of max pre-activation x weight), 'mean' (vertex mean)
  mmode   'none' (T75) or 'margin': at step 3 compare a gate's impact with the smallest |logit margin| of the
          frozen-readout over the cell's vertices instead of the surely-on scale (margin-aware impact, untried by T75)
Charge (`ops`): the T75 strict convention re-applied by this code (for equal-convention variant comparisons).
"""
import math

SQRT = math.sqrt
ATAN2 = math.atan2
PI = math.pi


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def unit(p, sqrt=SQRT):
    n = sqrt(dot(p, p))
    return (p[0] / n, p[1] / n, p[2] / n)


class Charge:
    def __init__(self):
        self.n = 0

    def add(self, k):
        self.n += k


def clip(poly, n, ch, sqrt=SQRT):
    s = [dot(n, v) for v in poly]
    ch.add(4 * len(poly))
    if min(s) >= 0:
        return poly, []
    if max(s) <= 0:
        return [], poly
    ins, out = [], []
    m = len(poly)
    for i in range(m):
        si, sj = s[i], s[(i + 1) % m]
        if si >= 0:
            ins.append(poly[i])
        if si <= 0:
            out.append(poly[i])
        if (si > 0 and sj < 0) or (si < 0 and sj > 0):
            a, b = abs(si), abs(sj)
            vj, vi = poly[(i + 1) % m], poly[i]
            p = unit((a * vj[0] + b * vi[0], a * vj[1] + b * vi[1], a * vj[2] + b * vi[2]), sqrt)
            ch.add(13)
            ins.append(p); out.append(p)
    return (ins if len(ins) >= 3 else []), (out if len(out) >= 3 else [])


def frac(poly, ch, atan2=ATAN2):
    if len(poly) < 3:
        return 0.0
    a = poly[0]; tot = 0.0
    for i in range(1, len(poly) - 1):
        b, c = poly[i], poly[i + 1]
        tot = tot + 2.0 * atan2(abs(dot(a, cross(b, c))), 1.0 + dot(a, b) + dot(a, c) + dot(b, c))
        ch.add(23)
    ch.add(1)
    return tot / (4 * PI)


# --------------------------------------------------------------------------- own half-chambers (model-free)
def _e(i, s=1.0):
    v = [0.0, 0.0, 0.0]; v[i] = s; return tuple(v)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _halves():
    import itertools
    import numpy as np
    out = []
    dummy = Charge()
    for sigma in (1.0, -1.0):
        for (mx, md, mn) in itertools.permutations(range(3)):
            # chamber x_mx > x_md > x_mn, sigma x0 > 0; class = md
            normals = [_sub(_e(mx), _e(md)), _sub(_e(md), _e(mn))]
            R = np.linalg.inv(np.array([normals[0], normals[1], _e(0, sigma)]))
            # start from the simplicial cone of the two order planes and the chart plane only if it is pointed;
            # always valid here because the three normals are independent
            poly = [tuple(R[:, m] / np.linalg.norm(R[:, m])) for m in range(3)]
            # the chart plane may be redundant / the cone {n1,n2,chart} equals chamber n chart (3 facets) -> fine
            split = tuple(_e(mx)[i] + _e(mn)[i] - 2 * _e(md)[i] for i in range(3))   # >0: md closer to mn
            for near, n_true in ((mx, _sub(_e(mx), _e(md))), (mn, _sub(_e(md), _e(mn)))):
                # half next to x_md = x_mx: x_mx - x_md < x_md - x_mn  <=>  split . x < 0
                nn = tuple(-x for x in split) if near == mx else split
                p, _ = clip(poly, nn, dummy)
                out.append(dict(sigma=sigma, truth=md, n_true=n_true, poly=p))
    return out


HALVES = _halves()


# ------------------------------------------------------------------------------------------- the recipe
def _resolve(poly, rows, n_true, w, eps, ch, smode="max", margin=None):
    on = [v for v in poly if abs(dot(n_true, v)) < 1e-9]
    ch.add(4 * len(poly))
    base = on if len(on) >= 2 else poly
    ref = tuple(sum(v[i] for v in base) / len(base) for i in range(3)); ch.add(3 * len(poly))
    g = [0.0] * len(rows); cut = []; scale = 0.0
    for k, a in enumerate(rows):
        s = [dot(a, v) for v in poly]; ch.add(4 * len(poly))
        if min(s) >= 0:
            g[k] = 1.0
            sv = max(s) if smode == "max" else sum(s) / len(s)
            scale = scale + sv * w[k]; ch.add(1 + len(s))
        elif max(s) <= 0:
            g[k] = 0.0
        else:
            g[k] = 1.0 if dot(a, ref) > 0 else 0.0; ch.add(4)
            depth = max(s) if g[k] == 0.0 else -min(s); ch.add(len(s))
            cut.append((depth * w[k], k)); ch.add(1)
    if margin is not None:            # margin-aware: compare with the smallest frozen logit margin in the cell
        scale = margin(g)
    chosen = [k for imp, k in cut if imp >= eps * scale]; ch.add(4 * len(cut))
    pieces = [(poly, g)]
    for k in chosen:
        nxt = []
        for p, gg in pieces:
            ins, outp = clip(p, rows[k], ch)
            if ins:
                g1 = list(gg); g1[k] = 1.0; nxt.append((ins, g1))
            if outp:
                g0 = list(gg); g0[k] = 0.0; nxt.append((outp, g0))
        pieces = nxt
    return pieces, len(cut), len(chosen)


def recipe(model, ch=None, eps=0.1, wmode="max", smode="max", mmode="none", stats=None):
    """model = (Wi list[4], Wh list[4][4], Wo list[3][4]) of floats or Num. Returns the accuracy estimate."""
    Wi, Wh, Wo = model
    ch = ch if ch is not None else Charge()
    H = len(Wi)
    e2, e3 = eps if isinstance(eps, tuple) else (eps, eps)
    err = 0.0
    for h in HALVES:
        sigma, truth, n_true = h["sigma"], h["truth"], h["n_true"]
        g1 = [1.0 if sigma * Wi[j] > 0 else 0.0 for j in range(H)]; ch.add(H)
        # step-2 pre-activation rows: a_k . x = (sum_j Wh[k][j] g1_j Wi_j) x0 + Wi_k x1
        c0 = [sum(Wh[k][j] * Wi[j] for j in range(H) if g1[j]) for k in range(H)]; ch.add(H * int(sum(g1)))
        A2 = [(c0[k], Wi[k], 0.0) for k in range(H)]
        if wmode == "max":
            w2 = [max(abs(Wh[j][k]) for j in range(H)) for k in range(H)]
            w3 = [max(abs(Wo[truth][k] - Wo[o][k]) for o in range(3)) for k in range(H)]
        elif wmode == "l2":
            w2 = [SQRT(sum(Wh[j][k] * Wh[j][k] for j in range(H))) for k in range(H)]
            w3 = [SQRT(sum((Wo[truth][k] - Wo[o][k]) ** 2 for o in range(3))) for k in range(H)]
        else:
            w2 = [1.0] * H; w3 = [1.0] * H
        ch.add(28)
        p2s, c, r = _resolve(h["poly"], A2, n_true, w2, e2, ch, smode)
        if stats is not None:
            stats["cut"] += c; stats["res"] += r
        for p2, g2 in p2s:
            A3 = [(sum(Wh[k][j] * A2[j][0] for j in range(H) if g2[j]), sum(Wh[k][j] * A2[j][1] for j in range(H) if g2[j]), Wi[k])
                  for k in range(H)]
            ch.add(4 * int(sum(g2)) * 2)
            margin = None
            if mmode == "margin":
                def margin(g3, p2=p2, A3=A3):
                    # smallest |l_truth - l_o| over the cell's vertices for the frozen gate pattern (no extra
                    # weighting: compared directly with impact = depth x |Wo[truth,k] - Wo[o,k]|)
                    best = None
                    for o in range(3):
                        if o == truth:
                            continue
                        d = tuple(sum((Wo[truth][k] - Wo[o][k]) * A3[k][q] for k in range(H) if g3[k]) for q in range(3))
                        for v in p2:
                            m_ = abs(dot(d, v))
                            best = m_ if best is None or m_ < best else best
                    ch.add(2 * (3 * H * 3 + 4 * len(p2)))
                    return best
            p3s, c, r = _resolve(p2, A3, n_true, w3, e3, ch, smode, margin)
            if stats is not None:
                stats["cut"] += c; stats["res"] += r; stats["cells"] += len(p3s)
            for p3, g3 in p3s:
                if sum(g3) == 0:
                    if truth != 0:
                        err = err + frac(p3, ch)
                    continue
                L = [tuple(sum(Wo[o][k] * A3[k][q] for k in range(H) if g3[k]) for q in range(3)) for o in range(3)]
                ch.add(3 * int(sum(g3)) * 3)
                q = p3
                for o in range(3):
                    if o == truth:
                        continue
                    ins, outp = clip(q, _sub(L[truth], L[o]), ch); ch.add(3)
                    if outp:
                        err = err + frac(outp, ch)
                    q = ins
                    if not q:
                        break
    return 1.0 - err


def as_lists(model):
    Wi, Wh, Wo = model
    return [float(x) for x in Wi], [[float(x) for x in r] for r in Wh], [[float(x) for x in r] for r in Wo]

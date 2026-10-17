"""T73: fibre-traced wedge estimator for bias-free 2nd-argmax RNNs, T = 3, any H (tiers F0 / F1 / FX).

Geometry: fiber3.FAMS, an exact model-independent partition of the sphere into fibres normal to the true boundary
circles x_i = x_j (each fibre runs from the circle to the equidistance plane; area element cos(phi) dphi dtheta).

Mechanism / estimator (deductive; every number is a formula of the weights):
  1. trace_arc: the gate pattern ON each true boundary arc, exactly (closed-form roots of A_t,k . p(theta)).
  2. On the fibre through p(theta), x = p + u n' (u = tan phi), the network is piecewise linear in u. It changes
     linear piece where a gate row crosses zero, u_g(theta) = -(A.p)/(A.n'), and its class where two logits tie,
     u_cd(theta) = -(w.p)/(w.n') with w = L_c - L_d. All are trig-linear in theta (alpha cos + beta sin), so the
     mass of a wrong fibre interval over a theta-interval of fixed combinatorics is a closed form (asin).
  3. Tiers:  F0 = pattern frozen at its arc value along the whole fibre (no gate off the arc resolved);
             F1 = fibre traced exactly through gate switches until the end of the first linear piece whose far
                  end is correctly classified ("the error is a wedge attached to the boundary");
             FX = traced to the clamp (exact; validates the geometry; = local enumeration of every region).
  4. Each trace returns its exact theta-validity interval (nearest roots of all order constraints between the
     breakpoint functions it compared), so the arc is split exactly where the combinatorics change.
accuracy = 1 - (sum of wrong masses) / (4 pi).
"""
import math

import numpy as np

from fiber3 import FAMS, Ops, _Sint

TWO_PI = 2 * math.pi


def rows_at(model, x, ops):
    """Pre-activation rows of all steps at point x (exact pattern), stacked (T*H, T); logit map L; live flag."""
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]
    A = np.zeros((H, T)); A[:, 0] = Wi
    R = []
    for t in range(T):
        if t > 0:
            A = Wh @ Ag; A[:, t] += Wi; ops.add(H * int(g.sum()) * t)
        R.append(A)
        g = A @ x > 0; ops.add(H * (t + 1))
        Ag = A * g[:, None]
    L = Wo @ Ag; ops.add(T * int(g.sum()) * T)
    return np.vstack(R), L, bool(g.any())


class State:
    """A linear piece seen from one fibre family: rows/logits and their dot products with P, U, n'."""
    __slots__ = ("R", "L", "live", "gal", "gbe", "ok", "LP", "LU", "Ln")

    def __init__(self, model, x, P, U, nn, ops):
        R, L, live = rows_at(model, x, ops)
        self.R, self.L, self.live = R, L, live
        RP, RU, Rn = R @ P, R @ U, R @ nn; ops.add(3 * 3 * len(R))
        ok = np.abs(Rn) > 1e-300
        with np.errstate(divide="ignore", invalid="ignore"):
            self.gal = np.where(ok, -RP / Rn, 0.0); self.gbe = np.where(ok, -RU / Rn, 0.0)
        ops.add(2 * len(R))
        self.ok = ok
        self.LP, self.LU, self.Ln = L @ P, L @ U, L @ nn; ops.add(3 * 3 * L.shape[0])


def _roots_vec(al, be, t0, t1):
    """For each (al, be): the root of al cos t + be sin t in (t0, t1) or nan (at most one: t1 - t0 <= pi)."""
    r = np.arctan2(al, -be) % math.pi
    r = np.where(r <= t0, r + math.pi, r)
    return np.where((r > t0) & (r < t1), r, np.nan)


class Cons:
    """Order constraints sign(f - g) that must hold on the validity interval (accumulated as coefficient rows)."""

    def __init__(self):
        self.al = []; self.be = []

    def add(self, al, be):
        self.al.append(np.atleast_1d(al)); self.be.append(np.atleast_1d(be))

    def interval(self, tm, t0, t1, ops):
        if not self.al:
            return t0, t1
        al = np.concatenate(self.al); be = np.concatenate(self.be)
        keep = (np.abs(al) + np.abs(be)) > 1e-13
        al, be = al[keep], be[keep]
        ops.add(6 * len(al))
        # roots of al cos + be sin: theta = atan2(al, -be) + k pi; nearest below / above tm
        r = np.arctan2(al, -be) % math.pi
        dl = (tm - r) % math.pi; dl = np.where(dl == 0, math.pi, dl)
        du = (r - tm) % math.pi; du = np.where(du == 0, math.pi, du)
        a = max(t0, tm - float(dl.min())) if len(al) else t0
        b = min(t1, tm + float(du.min())) if len(al) else t1
        return a, b


def fibre_trace(model, st0, P, U, nn, truth, fcl, theta, mode, ops, cache):
    """Wrong intervals on the fibre at theta, and the theta-interval on which they keep this combinatorics.
    Returns (intervals [(lo, hi)] with fn = (alpha, beta) or None for u = 0, (a, b))."""
    c, s = math.cos(theta), math.sin(theta)
    p = c * P + s * U
    ucl = fcl[0] * c + fcl[1] * s
    st = st0; f_u = None; u = 0.0
    out = []; cons = Cons()
    T = st.L.shape[0]
    for _ in range(64):
        # gate candidates of this piece
        ug = st.gal * c + st.gbe * s; ops.add(2 * len(ug))
        if mode == "F0":
            kk = -1; unext = math.inf
        else:
            if f_u is None:
                ahead = st.ok & (ug > 1e-15)
                cons.add(st.gal[st.ok], st.gbe[st.ok])                  # sign(u_g - 0)
            else:
                dal = st.gal - f_u[0]; dbe = st.gbe - f_u[1]
                same = st.ok & (np.abs(dal) + np.abs(dbe) < 1e-12 * (1 + abs(f_u[0]) + abs(f_u[1])))
                ahead = st.ok & ~same & (ug > u)
                m = st.ok & ~same
                cons.add(dal[m], dbe[m])                                 # sign(u_g - u_cur)
            if ahead.any():
                idx = np.flatnonzero(ahead); kk = int(idx[np.argmin(ug[idx])]); unext = float(ug[kk])
                if mode != "F1a":
                    o = idx[idx != kk]
                    cons.add(st.gal[o] - st.gal[kk], st.gbe[o] - st.gbe[kk])  # kk stays the nearest
                    cons.add(st.gal[kk] - fcl[0], st.gbe[kk] - fcl[1])        # vs clamp
            else:
                kk = -1; unext = math.inf
        if unext >= ucl:
            kk = -1; unext = ucl; f_next = fcl
        else:
            f_next = (float(st.gal[kk]), float(st.gbe[kk]))
        # class structure on (u, unext): logits a + v b
        a = st.LP * c + st.LU * s; b = st.Ln; ops.add(2 * T)
        edges = [(u, f_u)]
        if st.live:
            for d in range(T):
                if d == truth:
                    continue
                den = b[d] - b[truth]
                if abs(den) < 1e-300:
                    cons.add(st.LP[truth] - st.LP[d], st.LU[truth] - st.LU[d]); continue
                fd = ((st.LP[truth] - st.LP[d]) / den, (st.LU[truth] - st.LU[d]) / den); ops.add(4)
                for g in (f_u, f_next):
                    if g is None:
                        cons.add(fd[0], fd[1])
                    else:
                        cons.add(fd[0] - g[0], fd[1] - g[1])
                v = fd[0] * c + fd[1] * s
                if u < v < unext:
                    edges.append((v, fd))
            # the order of the in-range crossings among themselves
            if len(edges) > 2:
                for x in range(1, len(edges)):
                    for y in range(x + 1, len(edges)):
                        cons.add(edges[x][1][0] - edges[y][1][0], edges[x][1][1] - edges[y][1][1])
        edges.sort(key=lambda e: e[0])
        edges.append((unext, f_next))
        if mode == "F1a":
            # attached wedge only: wrong from u, traced to the first edge where the class becomes the truth
            stop = None
            for q in range(len(edges) - 1):
                vm = 0.5 * (edges[q][0] + edges[q + 1][0])
                cls = int(np.argmax(a + vm * b)) if st.live else 0; ops.add(2 * T)
                if cls == truth:
                    stop = q; break
            if stop == 0:
                break                                                   # correct right at u: nothing attached
            if stop is not None:
                f_end = edges[stop][1]
                if kk >= 0:                                             # gates ahead stay beyond the end
                    m = ahead
                    cons.add(st.gal[m] - f_end[0], st.gbe[m] - f_end[1])
                cons.add(fcl[0] - f_end[0], fcl[1] - f_end[1])
                out.append((f_u, f_end) if not (out and out[-1][1] is f_u) else (out.pop()[0], f_end))
                break
            if kk >= 0:
                o = np.flatnonzero(ahead); o = o[o != kk]
                cons.add(st.gal[o] - st.gal[kk], st.gbe[o] - st.gbe[kk])
            cons.add(f_next[0] - fcl[0], f_next[1] - fcl[1]) if kk >= 0 else None
            out.append((f_u, f_next) if not (out and out[-1][1] is f_u) else (out.pop()[0], f_next))
            if kk < 0:
                break
            key = (id(st), kk)
            if key not in cache:
                x = p + (unext * (1 + 1e-9) + 1e-12) * nn
                cache[key] = State(model, x, P, U, nn, ops)
            st = cache[key]; u = unext; f_u = f_next
            continue
        cls_end = truth
        for q in range(len(edges) - 1):
            vm = 0.5 * (edges[q][0] + edges[q + 1][0])
            cls = int(np.argmax(a + vm * b)) if st.live else 0; ops.add(2 * T)
            if cls != truth:
                if out and out[-1][1] is edges[q][1]:
                    out[-1] = (out[-1][0], edges[q + 1][1])
                else:
                    out.append((edges[q][1], edges[q + 1][1]))
            cls_end = cls
        if kk < 0 or (mode == "F1" and cls_end == truth):
            break
        # step across gate kk; the pattern beyond it is cached per (piece, gate) within this call
        key = (id(st), kk)
        if key not in cache:
            x = p + (unext * (1 + 1e-9) + 1e-12) * nn
            cache[key] = State(model, x, P, U, nn, ops)
        st = cache[key]; u = unext; f_u = f_next
    return out, cons


def _integrate(model, st0, P, U, nn, truth, fcl, t0, t1, mode, ops, cache, budget):
    tot = 0.0
    stack = [(t0, t1)]
    while stack:
        a0, a1 = stack.pop()
        if a1 - a0 < 1e-13:
            continue
        tm = 0.5 * (a0 + a1)
        iv, cons = fibre_trace(model, st0, P, U, nn, truth, fcl, tm, mode, ops, cache)
        a, b = cons.interval(tm, a0, a1, ops)
        budget[0] += 1
        if budget[0] > 200000:
            a, b = a0, a1
        for lo, hi in iv:
            tot += _Sint(hi[0], hi[1], a, b); ops.add(12)
            if lo is not None:
                tot -= _Sint(lo[0], lo[1], a, b); ops.add(12)
        if a > a0:
            stack.append((a0, a))
        if b < a1:
            stack.append((b, a1))
    return tot


def trace_arc_rows(model, P, U, lo, hi, ops):
    """Arc sub-intervals (t0, t1, x_mid) on which the gate pattern is constant; all gate switches on the arc are
    found exactly, step by step (roots of the step-t rows inside each step-(t-1) sub-arc)."""
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]
    segs = [(lo, hi)]
    for t in range(T):
        nxt = []
        for (a0, a1) in segs:
            tm = 0.5 * (a0 + a1)
            R, _, _ = rows_at(model, math.cos(tm) * P + math.sin(tm) * U, ops)
            A = R[t * H:(t + 1) * H]
            r = _roots_vec(A @ P, A @ U, a0, a1); ops.add(2 * H * T + 3 * H)
            bs = [a0] + sorted(float(v) for v in r if not np.isnan(v)) + [a1]
            nxt += [(bs[q], bs[q + 1]) for q in range(len(bs) - 1)]
        segs = nxt
    return [(a0, a1, math.cos(0.5 * (a0 + a1)) * P + math.sin(0.5 * (a0 + a1)) * U) for a0, a1 in segs]


def est_fibre(model, mode="F1", ops=None, info=None):
    ops = ops if ops is not None else Ops()
    tot = 0.0; by = {}; nseg = 0; budget = [0]
    for f in FAMS:
        P, U = f["P"], f["U"]
        segs = trace_arc_rows(model, P, U, 0.0, math.pi, ops)
        nseg += len(segs)
        for nn, truth, wcl in f["sides"]:
            wn = -(wcl @ nn); fcl = ((wcl @ P) / wn, (wcl @ U) / wn); ops.add(11)
            e = 0.0
            for t0, t1, xm in segs:
                cache = {}
                st0 = State(model, xm, P, U, nn, ops)
                e += _integrate(model, st0, P, U, nn, truth, fcl, t0, t1, mode, ops, cache, budget)
            tot += e
            by[(f["circle"], f["s"], truth)] = e / (4 * math.pi)
    if info is not None:
        info["err_by"] = by; info["segments"] = nseg; info["traces"] = budget[0]
    return 1.0 - tot / (4 * math.pi)


# ----------------------------------------------------------------------------- tier F2: attached wedges at both ends
def _walk(model, st, P, U, nn, truth, theta, f_start, u_start, sgn, f_bound, u_bound, ops, cache, cons):
    """Walk the fibre from u_start in direction sgn (+1 up, -1 down) while it is misclassified (attached-wedge
    semantics), stopping at the first correctly classified point or at u_bound. Returns (intervals, f_stop, u_stop)
    with intervals as (lo_fn, hi_fn) in increasing u; fn None means u = 0."""
    c, s = math.cos(theta), math.sin(theta)
    p = c * P + s * U
    T = st.L.shape[0]
    f_u, u = f_start, u_start
    out = []
    zero = (0.0, 0.0)

    def fv(f):
        return zero if f is None else f
    for _ in range(64):
        ug = st.gal * c + st.gbe * s; ops.add(2 * len(ug))
        fu = fv(f_u)
        dal = st.gal - fu[0]; dbe = st.gbe - fu[1]
        same = st.ok & (np.abs(dal) + np.abs(dbe) < 1e-12 * (1 + abs(fu[0]) + abs(fu[1])))
        m = st.ok & ~same
        cons.add(dal[m], dbe[m])
        ahead = m & ((ug - u) * sgn > 0) & ((ug - u_bound) * sgn < 0)
        fb = fv(f_bound)
        if ahead.any():
            idx = np.flatnonzero(ahead); kk = int(idx[np.argmin(ug[idx] * sgn)])
            unext = float(ug[kk]); f_next = (float(st.gal[kk]), float(st.gbe[kk]))
            cons.add(st.gal[kk] - fb[0], st.gbe[kk] - fb[1])
        else:
            kk = -1; unext = u_bound; f_next = f_bound
            o = np.flatnonzero(m & ((ug - u) * sgn > 0))
            cons.add(st.gal[o] - fb[0], st.gbe[o] - fb[1])
        a = st.LP * c + st.LU * s; b = st.Ln; ops.add(2 * T)
        lo_u, hi_u = (u, unext) if sgn > 0 else (unext, u)
        edges = []
        if st.live:
            for d in range(T):
                if d == truth:
                    continue
                den = b[d] - b[truth]
                if abs(den) < 1e-300:
                    cons.add(st.LP[truth] - st.LP[d], st.LU[truth] - st.LU[d]); continue
                fd = ((st.LP[truth] - st.LP[d]) / den, (st.LU[truth] - st.LU[d]) / den); ops.add(4)
                fn = fv(f_next)
                cons.add(fd[0] - fu[0], fd[1] - fu[1]); cons.add(fd[0] - fn[0], fd[1] - fn[1])
                v = fd[0] * c + fd[1] * s
                if lo_u < v < hi_u:
                    edges.append((v, fd))
            if len(edges) > 1:
                for x in range(len(edges)):
                    for y in range(x + 1, len(edges)):
                        cons.add(edges[x][1][0] - edges[y][1][0], edges[x][1][1] - edges[y][1][1])
        edges.sort(key=lambda e: e[0] * sgn)
        seq = [(u, f_u)] + edges + [(unext, f_next)]
        stop = None
        for q in range(len(seq) - 1):
            vm = 0.5 * (seq[q][0] + seq[q + 1][0])
            cls = int(np.argmax(a + vm * b)) if st.live else 0; ops.add(2 * T)
            if cls == truth:
                stop = q; break
        if stop is not None:
            if stop > 0:
                fe = seq[stop][1]
                if kk >= 0:          # the gate ahead stays beyond the stop point
                    cons.add(st.gal[kk] - fv(fe)[0], st.gbe[kk] - fv(fe)[1])
                out.append((f_start, fe) if sgn > 0 else (fe, f_start))
                return out, fe, seq[stop][0]
            return out, f_u, u if not out else None
        if kk < 0:
            out.append((f_start, f_bound) if sgn > 0 else (f_bound, f_start))
            return out, f_bound, u_bound
        key = (id(st), kk)
        if key not in cache:
            x = p + (unext + sgn * (abs(unext) * 1e-9 + 1e-12)) * nn
            cache[key] = State(model, x, P, U, nn, ops)
        st = cache[key]; u = unext; f_u = f_next
    out.append((f_start, f_u) if sgn > 0 else (f_u, f_start))
    return out, f_u, u


def trace_F2(model, st0, P, U, nn, truth, fcl, theta, ops, cache):
    cons = Cons()
    c, s = math.cos(theta), math.sin(theta)
    ucl = fcl[0] * c + fcl[1] * s
    if ucl <= 0:
        return [], cons
    iv, f_a, u_a = _walk(model, st0, P, U, nn, truth, theta, None, 0.0, 1, fcl, ucl, ops, cache, cons)
    if iv and iv[-1][1] is fcl:
        return iv, cons                                  # wrong all the way to the clamp
    if u_a is None:
        f_a, u_a = None, 0.0
    # reverse walk from the clamp, state just below it
    p = c * P + s * U
    x = p + ucl * (1 - 1e-9) * nn
    stc = State(model, x, P, U, nn, ops)
    iv2, _, _ = _walk(model, stc, P, U, nn, truth, theta, fcl, ucl, -1, f_a, u_a, ops, cache, cons)
    return iv + iv2, cons


def _integrate2(model, st0, P, U, nn, truth, fcl, t0, t1, ops, cache, budget):
    tot = 0.0
    stack = [(t0, t1)]
    while stack:
        a0, a1 = stack.pop()
        if a1 - a0 < 1e-13:
            continue
        tm = 0.5 * (a0 + a1)
        iv, cons = trace_F2(model, st0, P, U, nn, truth, fcl, tm, ops, cache)
        a, b = cons.interval(tm, a0, a1, ops)
        budget[0] += 1
        if budget[0] > 400000:
            a, b = a0, a1
        for lo, hi in iv:
            if hi is not None:
                tot += _Sint(hi[0], hi[1], a, b); ops.add(12)
            if lo is not None:
                tot -= _Sint(lo[0], lo[1], a, b); ops.add(12)
        if a > a0:
            stack.append((a0, a))
        if b < a1:
            stack.append((b, a1))
    return tot


def est_fibre2(model, ops=None, info=None):
    """Tier F2: attached wedges traced exactly from the boundary (u = 0 up) and from the clamp (down)."""
    ops = ops if ops is not None else Ops()
    tot = 0.0; by = {}; nseg = 0; budget = [0]
    for f in FAMS:
        P, U = f["P"], f["U"]
        segs = trace_arc_rows(model, P, U, 0.0, math.pi, ops)
        nseg += len(segs)
        for nn, truth, wcl in f["sides"]:
            wn = -(wcl @ nn); fcl = ((wcl @ P) / wn, (wcl @ U) / wn); ops.add(11)
            e = 0.0
            for t0, t1, xm in segs:
                cache = {}
                st0 = State(model, xm, P, U, nn, ops)
                e += _integrate2(model, st0, P, U, nn, truth, fcl, t0, t1, ops, cache, budget)
            tot += e
            by[(f["circle"], f["s"], truth)] = e / (4 * math.pi)
    if info is not None:
        info["err_by"] = by; info["segments"] = nseg; info["traces"] = budget[0]
    return 1.0 - tot / (4 * math.pi)

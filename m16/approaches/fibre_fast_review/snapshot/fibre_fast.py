"""T95: the frozen F2L2 fibre recipe (fibre.py, LOOK = 2) with exact cost reductions; same output to rounding.

Reductions (all exact, none changes what is computed):
  R1 gate pruning: on an arc segment (t0, t1) the crossing point of gate g along every fibre is the trig-linear
     u_g(theta) = alpha cos + beta sin; its exact range over the segment is computed once per piece state. A gate
     whose range misses (0, max clamp) never switches on any fibre of the segment, so it is never tested and needs
     no order constraint ("predict from the weights which gates can switch along a fibre").
  R2 range-filtered constraints: an order constraint sign(f - g) is only kept if the ranges of f and g over the
     segment overlap (otherwise its sign cannot change).
  R3 incremental pieces: crossing gate (step t, neuron k) at x* changes the later rows only by rank-1 terms
     c_r (x) a, a = A_t,k (a . x* = 0, so later pre-activations are continuous and their gates unchanged):
     c_{t+1} = +-Wh[:, k], c_{r+1} = Wh (g_r * c_r), Delta L = Wo (g_{T-1} * c_{T-1}) (x) a.
  R4 both fibre sides share the arc state (the n-dots change sign); the clamp state of the reverse walk is reused
     when a cheap self-consistency check (all its gate signs hold at the new clamp point) passes.
Reference path: fibre.est_fibre2 with fibre.LOOK = 2 (verify_fast.py checks equality on all scored models).
"""
import math

import numpy as np

from fiber3 import FAMS, Ops, _Sint
from fibre import Cons, rows_at, trace_arc_rows

LOOK = 2
CLAMP_TRIES = 2
XCROSS = [None]; CTXV = [None]
PRUNE = True; RANGEF = True; INCR = True; CLAMPC = True; FLIP = True; SHARE = True   # switches (debug / ablation)


def _range(al, be, t0, t1):
    """Exact min/max of al cos t + be sin t over [t0, t1] (vectorised; t1 - t0 < pi)."""
    v0 = al * math.cos(t0) + be * math.sin(t0); v1 = al * math.cos(t1) + be * math.sin(t1)
    lo = np.minimum(v0, v1); hi = np.maximum(v0, v1)
    R = np.hypot(al, be); th = np.arctan2(be, al)            # max R at th (+2k pi), min -R at th + pi
    for c, sgn in ((th, 1.0), (th + math.pi, -1.0)):
        c = (c - t0) % (2 * math.pi) + t0
        inside = c < t1
        hi = np.where(inside & (sgn > 0), R, hi); lo = np.where(inside & (sgn < 0), -R, lo)
    return lo, hi


class Piece:
    """One linear piece seen from one fibre side: rows, gates, logits, dots with (P, U, n'), and the candidate
    gates of the current segment (R1)."""
    __slots__ = ("R", "G", "L", "live", "RP", "RU", "Rn", "LP", "LU", "Ln", "gal", "gbe", "ok", "kids", "segd", "cur", "parent", "ch")

    def finish(self, seg, ops):
        """Segment-independent derived quantities (crossing functions)."""
        ok = np.abs(self.Rn) > 1e-300
        with np.errstate(divide="ignore", invalid="ignore"):
            self.gal = np.where(ok, -self.RP / self.Rn, 0.0); self.gbe = np.where(ok, -self.RU / self.Rn, 0.0)
        ops.add(2 * len(ok))
        self.ok = ok
        self.kids = {}; self.segd = {}; self.parent = None; self.ch = None

    def use(self, seg, ops):
        """Select segment seg: ranges and candidates (R1) computed once per (piece, segment) and cached (R5)."""
        d = self.segd.get(seg)
        if d is None:
            par = getattr(self, "parent", None)
            pd = par.segd.get(seg) if par is not None else None
            if pd is not None:                      # inherit the unchanged rows' ranges from the parent piece
                lo, hi, cand = pd[0].copy(), pd[1].copy(), pd[2].copy(); ch = self.ch
                if len(ch):
                    l2, h2 = _range(self.gal[ch], self.gbe[ch], seg[0], seg[1]); ops.add(12 * len(ch))
                    lo[ch] = l2; hi[ch] = h2
                    cand[ch] = self.ok[ch] & (((h2 > 0) & (l2 < seg[2])) | (not PRUNE)); ops.add(2 * len(ch))
            else:
                lo, hi = _range(self.gal, self.gbe, seg[0], seg[1]); ops.add(12 * len(self.ok))
                cand = self.ok & (((hi > 0) & (lo < seg[2])) | (not PRUNE)); ops.add(2 * len(self.ok))
            d = (lo, hi, cand); self.segd[seg] = d
        self.cur = d
        return self

    @property
    def rlo(self):
        return self.cur[0]

    @property
    def rhi(self):
        return self.cur[1]

    @property
    def cand(self):
        return self.cur[2]

    @staticmethod
    def at(model, x, P, U, nn, seg, ops):
        R, L, live = rows_at(model, x, ops)
        p = Piece()
        p.R, p.L, p.live = R, L, live
        p.G = (R @ x) > 0; ops.add(3 * len(R))
        p.RP, p.RU, p.Rn = R @ P, R @ U, R @ nn; ops.add(9 * len(R))
        p.LP, p.LU, p.Ln = L @ P, L @ U, L @ nn; ops.add(9 * L.shape[0])
        p.finish(seg, ops)
        return p.use(seg, ops)

    def flipped(self, seg, ops):
        """The same piece seen from the other fibre side (n' -> -n')."""
        p = Piece()
        p.R, p.L, p.live, p.G = self.R, self.L, self.live, self.G
        p.RP, p.RU, p.Rn = self.RP, self.RU, -self.Rn
        p.LP, p.LU, p.Ln = self.LP, self.LU, -self.Ln
        ops.add(len(self.Rn) + len(self.Ln))
        p.finish(seg, ops)
        return p.use(seg, ops)

    def _flip(self, model, grp, ops):
        """New piece with the gates grp flipped (rank-1 updates of the later rows and the logits, R3)."""
        Wi, Wh, Wo = model
        H = len(Wi); T = Wo.shape[0]
        p = Piece()
        p.G = self.G.copy(); p.R = self.R.copy(); p.RP = self.RP.copy(); p.RU = self.RU.copy(); p.Rn = self.Rn.copy()
        p.L = self.L.copy(); p.LP = self.LP.copy(); p.LU = self.LU.copy(); p.Ln = self.Ln.copy()
        tmin = T
        for q in sorted(int(v) for v in grp):
            t, k = divmod(q, H)
            tmin = min(tmin, t)
            s = -1.0 if p.G[q] else 1.0
            a = p.R[q].copy(); aP, aU, an = p.RP[q], p.RU[q], p.Rn[q]
            p.G[q] = not p.G[q]
            if t == T - 1:
                cL = s * Wo[:, k]
            else:
                c = s * Wh[:, k]
                for r in range(t + 1, T):
                    b = slice(r * H, (r + 1) * H)
                    p.R[b] += np.outer(c, a); p.RP[b] += c * aP; p.RU[b] += c * aU; p.Rn[b] += c * an
                    ops.add(6 * H)
                    gr = p.G[b]
                    if r == T - 1:
                        cL = Wo @ (gr * c); ops.add(T * int(gr.sum()))
                    else:
                        c = Wh @ (gr * c); ops.add(H * int(gr.sum()))
            p.L += np.outer(cL, a); p.LP += cL * aP; p.LU += cL * aU; p.Ln += cL * an
            ops.add(6 * T)
        p.live = bool(p.G[(T - 1) * H:].any())
        p.gal = self.gal.copy(); p.gbe = self.gbe.copy(); p.ok = self.ok.copy()
        ch = np.arange((tmin + 1) * H, T * H) if tmin < T else np.arange(0)
        if len(ch):
            okc = np.abs(p.Rn[ch]) > 1e-300
            with np.errstate(divide="ignore", invalid="ignore"):
                p.gal[ch] = np.where(okc, -p.RP[ch] / p.Rn[ch], 0.0); p.gbe[ch] = np.where(okc, -p.RU[ch] / p.Rn[ch], 0.0)
            p.ok[ch] = okc; ops.add(2 * len(ch))
        p.segd = {}; p.kids = {}; p.parent = self; p.ch = ch
        return p

    def moved(self, model, x, ops):
        """Piece of the pattern at point x, reached from this piece by flipping the earliest-step gates whose sign
        differs at x, repeatedly (each flip is exact; later rows are re-tested after each flip)."""
        H = len(model[0])
        p = self
        for _ in range(4 * len(self.G)):
            z = p.R @ x; ops.add(3 * len(z))
            D = np.flatnonzero((z > 0) != p.G)
            if len(D) == 0:
                return p
            t = D.min() // H
            p = p._flip(model, D[D // H == t], ops)
        raise RuntimeError("moved: no convergence")

    def child(self, model, kk, seg, ops):
        """Piece beyond the crossing of gate kk (R3, rank-1 updates). Every gate with the same crossing function
        (e.g. all step-1 gates share the plane x0 = 0) flips with it."""
        if kk in self.kids:
            return self.kids[kk].use(seg, ops)
        if not INCR:
            x = XCROSS[0]
            p = Piece.at(model, x, CTXV[0].P, CTXV[0].U, CTXV[0].nn, seg, ops)
            self.kids[kk] = p
            return p.use(seg, ops)
        Wi, Wh, Wo = model
        tol = 1e-12 * (1 + abs(self.gal[kk]) + abs(self.gbe[kk]))
        grp = np.flatnonzero(self.ok & (np.abs(self.gal - self.gal[kk]) + np.abs(self.gbe - self.gbe[kk]) < tol))
        ops.add(3 * len(self.ok))
        p = self._flip(model, grp, ops)
        p.kids = {}
        self.kids[kk] = p
        return p.use(seg, ops)


def _frange(f, seg):
    lo, hi = _range(np.array([f[0]]), np.array([f[1]]), seg[0], seg[1])
    return float(lo[0]), float(hi[0])


class Ctx:
    """Per (family side, segment) context: geometry, clamp, segment bounds."""
    __slots__ = ("model", "P", "U", "nn", "truth", "fcl", "seg", "crng", "clamp_cache")


def _add_gate_cons(cons, st, idx, f, frng, ops):
    """Constraints sign(u_g - f) for gates idx whose segment range overlaps f's range (R2)."""
    if len(idx) == 0:
        return
    m = ((st.rhi[idx] >= frng[0]) & (st.rlo[idx] <= frng[1])) | (not RANGEF); ops.add(2 * len(idx))
    j = idx[m]
    if len(j):
        cons.add(st.gal[j] - f[0], st.gbe[j] - f[1])


def _class_fns(st, truth, T):
    out = {}
    for d in range(T):
        if d == truth:
            continue
        den = st.Ln[d] - st.Ln[truth]
        if abs(den) < 1e-300:
            out[d] = None
        else:
            out[d] = ((st.LP[truth] - st.LP[d]) / den, (st.LU[truth] - st.LU[d]) / den)
    return out


def forward_trace(cx, st0, theta, ops):
    """fibre.fibre_trace(mode='F1', look=LOOK) with R1-R3. Returns (intervals, cons, f_stop, u_stop)."""
    c, s = math.cos(theta), math.sin(theta)
    fcl = cx.fcl; ucl = fcl[0] * c + fcl[1] * s
    st = st0; f_u = None; u = 0.0; frng_u = (0.0, 0.0)
    out = []; cons = Cons(); n_ok = 0
    T = st.L.shape[0]; truth = cx.truth
    f_stop, u_stop = None, 0.0
    for _ in range(64):
        ci = np.flatnonzero(st.cand); ops.add(len(st.cand) // 8 + 1)
        ug = st.gal[ci] * c + st.gbe[ci] * s; ops.add(2 * len(ci))
        if f_u is None:
            ahead_m = ug > 1e-15
            _add_gate_cons(cons, st, ci, (0.0, 0.0), (0.0, 0.0), ops)
        else:
            dal = st.gal[ci] - f_u[0]; dbe = st.gbe[ci] - f_u[1]
            same = np.abs(dal) + np.abs(dbe) < 1e-12 * (1 + abs(f_u[0]) + abs(f_u[1]))
            ahead_m = ~same & (ug > u)
            _add_gate_cons(cons, st, ci[~same], f_u, frng_u, ops)
        ahead = ci[ahead_m]
        if len(ahead):
            uga = ug[ahead_m]; q = int(np.argmin(uga)); kk = int(ahead[q]); unext = float(uga[q])
            fk = (float(st.gal[kk]), float(st.gbe[kk])); rk = (float(st.rlo[kk]), float(st.rhi[kk]))
            _add_gate_cons(cons, st, ahead[ahead != kk], fk, rk, ops)
            cons.add(st.gal[kk] - fcl[0], st.gbe[kk] - fcl[1])
        else:
            kk = -1; unext = math.inf
        if unext >= ucl:
            kk = -1; unext = ucl; f_next = fcl; rng_next = cx.crng
        else:
            f_next = fk; rng_next = rk
        a = st.LP * c + st.LU * s; b = st.Ln; ops.add(2 * T)
        edges = [(u, f_u)]
        if st.live:
            for d, fd in _class_fns(st, truth, T).items():
                ops.add(4)
                if fd is None:
                    cons.add(st.LP[truth] - st.LP[d], st.LU[truth] - st.LU[d]); continue
                for g in (f_u, f_next):
                    if g is None:
                        cons.add(fd[0], fd[1])
                    else:
                        cons.add(fd[0] - g[0], fd[1] - g[1])
                v = fd[0] * c + fd[1] * s
                if u < v < unext:
                    edges.append((v, fd))
            if len(edges) > 2:
                for x in range(1, len(edges)):
                    for y in range(x + 1, len(edges)):
                        cons.add(edges[x][1][0] - edges[y][1][0], edges[x][1][1] - edges[y][1][1])
        edges.sort(key=lambda e: e[0])
        edges.append((unext, f_next))
        cls_end = truth; piece_ok = True
        for q in range(len(edges) - 1):
            vm = 0.5 * (edges[q][0] + edges[q + 1][0])
            cls = int(np.argmax(a + vm * b)) if st.live else 0; ops.add(2 * T)
            if cls != truth:
                piece_ok = False
                if out and out[-1][1] is edges[q][1]:
                    out[-1] = (out[-1][0], edges[q + 1][1])
                else:
                    out.append((edges[q][1], edges[q + 1][1]))
            cls_end = cls
        n_ok = n_ok + 1 if piece_ok else 0
        f_stop, u_stop = f_next, unext
        if kk < 0 or (cls_end == truth and (LOOK <= 1 or n_ok >= LOOK)):
            break
        XCROSS[0] = (c * cx.P + s * cx.U) + (unext * (1 + 1e-9) + 1e-12) * cx.nn; CTXV[0] = cx
        st = st.child(cx.model, kk, cx.seg, ops)
        u = unext; f_u = f_next; frng_u = rng_next
    return out, cons, f_stop, u_stop


def reverse_walk(cx, st, theta, f_bound, u_bound, cons, ops):
    """fibre._walk from the clamp downwards (attached-wedge semantics) with R1-R3."""
    c, s = math.cos(theta), math.sin(theta)
    T = st.L.shape[0]; truth = cx.truth
    f_start = cx.fcl; f_u, u = cx.fcl, cx.fcl[0] * c + cx.fcl[1] * s; frng_u = cx.crng
    zero = (0.0, 0.0)
    fv = lambda f: zero if f is None else f
    frng_b = (0.0, 0.0) if f_bound is None else _frange(f_bound, cx.seg)
    out = []
    for _ in range(64):
        ci = np.flatnonzero(st.cand); ops.add(len(st.cand) // 8 + 1)
        ug = st.gal[ci] * c + st.gbe[ci] * s; ops.add(2 * len(ci))
        dal = st.gal[ci] - f_u[0]; dbe = st.gbe[ci] - f_u[1]
        same = np.abs(dal) + np.abs(dbe) < 1e-12 * (1 + abs(f_u[0]) + abs(f_u[1]))
        _add_gate_cons(cons, st, ci[~same], f_u, frng_u, ops)
        ahead_m = ~same & (ug < u) & (ug > u_bound)
        fb = fv(f_bound)
        if ahead_m.any():
            ahead = ci[ahead_m]; uga = ug[ahead_m]; q = int(np.argmax(uga)); kk = int(ahead[q]); unext = float(uga[q])
            f_next = (float(st.gal[kk]), float(st.gbe[kk])); rng_next = (float(st.rlo[kk]), float(st.rhi[kk]))
            cons.add(st.gal[kk] - fb[0], st.gbe[kk] - fb[1])
        else:
            kk = -1; unext = u_bound; f_next = f_bound; rng_next = frng_b
            o = ci[~same & (ug < u)]
            _add_gate_cons(cons, st, o, fb, frng_b, ops)
        a = st.LP * c + st.LU * s; b = st.Ln; ops.add(2 * T)
        lo_u, hi_u = unext, u
        edges = []
        if st.live:
            for d, fd in _class_fns(st, truth, T).items():
                ops.add(4)
                if fd is None:
                    cons.add(st.LP[truth] - st.LP[d], st.LU[truth] - st.LU[d]); continue
                fn = fv(f_next)
                cons.add(fd[0] - f_u[0], fd[1] - f_u[1]); cons.add(fd[0] - fn[0], fd[1] - fn[1])
                v = fd[0] * c + fd[1] * s
                if lo_u < v < hi_u:
                    edges.append((v, fd))
            if len(edges) > 1:
                for x in range(len(edges)):
                    for y in range(x + 1, len(edges)):
                        cons.add(edges[x][1][0] - edges[y][1][0], edges[x][1][1] - edges[y][1][1])
        edges.sort(key=lambda e: -e[0])
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
                if kk >= 0:
                    cons.add(st.gal[kk] - fv(fe)[0], st.gbe[kk] - fv(fe)[1])
                out.append((fe, f_start))
                return out
            if f_u is f_start:
                return out
            out.append((f_u, f_start))
            return out
        if kk < 0:
            out.append((f_bound, f_start))
            return out
        XCROSS[0] = (c * cx.P + s * cx.U) + (unext - (abs(unext) * 1e-9 + 1e-12)) * cx.nn; CTXV[0] = cx
        st = st.child(cx.model, kk, cx.seg, ops)
        u = unext; f_u = f_next; frng_u = rng_next
    out.append((f_u, f_start))
    return out


def _clamp_piece(cx, theta, ops):
    """Piece just below the clamp at theta; a cached piece is reused if all its gate signs hold there (R4)."""
    c, s = math.cos(theta), math.sin(theta)
    p = c * cx.P + s * cx.U
    ucl = cx.fcl[0] * c + cx.fcl[1] * s
    x = p + ucl * (1 - 1e-9) * cx.nn
    cc = cx.clamp_cache if CLAMPC else []
    for q in range(min(len(cc), CLAMP_TRIES)):          # most recently used first
        st = cc[q]
        z = st.R @ x; ops.add(3 * len(z))
        if np.array_equal(z > 0, st.G):
            if q:
                cc.insert(0, cc.pop(q))
            return st.use(cx.seg, ops)
    st = cc[0].moved(cx.model, x, ops).use(cx.seg, ops) if (cc and SHARE) else Piece.at(cx.model, x, cx.P, cx.U, cx.nn, cx.seg, ops)
    cx.clamp_cache.insert(0, st)
    return st


def trace(cx, st0, theta, ops):
    c, s = math.cos(theta), math.sin(theta)
    ucl = cx.fcl[0] * c + cx.fcl[1] * s
    if ucl <= 0:
        return [], Cons()
    iv, cons, f_a, u_a = forward_trace(cx, st0, theta, ops)
    if f_a is cx.fcl:
        return iv, cons
    if iv and iv[-1][1] is cx.fcl:
        return iv, cons
    stc = _clamp_piece(cx, theta, ops)
    iv2 = reverse_walk(cx, stc, theta, f_a, u_a, cons, ops)
    return iv + iv2, cons


def _integrate(cx, st0, t0, t1, ops, stats):
    tot = 0.0
    stack = [(t0, t1)]
    while stack:
        a0, a1 = stack.pop()
        if a1 - a0 < 1e-13:
            continue
        tm = 0.5 * (a0 + a1)
        iv, cons = trace(cx, st0, tm, ops)
        a, b = cons.interval(tm, a0, a1, ops)
        stats["traces"] += 1; stats["cons"] += sum(len(x) for x in cons.al)
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


def est_fast(model, ops=None, info=None):
    ops = ops if ops is not None else Ops()
    stats = {"traces": 0, "cons": 0, "segments": 0}
    tot = 0.0; by = {}
    for f in FAMS:
        P, U = f["P"], f["U"]
        segs = trace_arc_rows(model, P, U, 0.0, math.pi, ops)
        stats["segments"] += len(segs)
        sides = []
        for nn, truth, wcl in f["sides"]:
            wn = -(wcl @ nn); fcl = ((wcl @ P) / wn, (wcl @ U) / wn); ops.add(11)
            sides.append((nn, truth, fcl))
        e_side = [0.0, 0.0]
        clamp_caches = [[], []]
        prev = None; flipcache = {}
        for t0, t1, xm in segs:
            for si, (nn, truth, fcl) in enumerate(sides):
                cx = Ctx()
                cx.model, cx.P, cx.U, cx.nn, cx.truth, cx.fcl = model, P, U, nn, truth, fcl
                crng = _frange(fcl, (t0, t1)); ops.add(12)
                cx.crng = crng; cx.seg = (t0, t1, crng[1]); cx.clamp_cache = clamp_caches[si]
                if si == 0:
                    if prev is None or not SHARE:
                        st0 = Piece.at(model, xm, P, U, nn, cx.seg, ops)
                    else:
                        st0 = prev.moved(model, xm, ops).use(cx.seg, ops)
                    prev = st0; base = st0
                else:
                    key = base.G.tobytes()
                    if key not in flipcache or not FLIP:
                        flipcache[key] = base.flipped(cx.seg, ops)
                    st0 = flipcache[key].use(cx.seg, ops)
                e_side[si] += _integrate(cx, st0, t0, t1, ops, stats)
        for si, (nn, truth, fcl) in enumerate(sides):
            by[(f["circle"], f["s"], truth)] = e_side[si] / (4 * math.pi)
        tot += sum(e_side)
    if info is not None:
        info.update(stats); info["err_by"] = by
    return 1.0 - tot / (4 * math.pi)

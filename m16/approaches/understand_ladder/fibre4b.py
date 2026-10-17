"""T100: the frozen F2L2 recipe at T = 4 (fibre4 geometry): forward trace from the boundary until LOOK = 2
consecutive linear pieces are fully correct, plus the attached-wedge walk back from the clamp (fibre.py's
trace_F2 / _walk ported to 2-D faces; breakpoint functions are vectors on the hyperplane basis, constraints are
great circles). Deductive; every coefficient from the weights."""
import math

import numpy as np

import fibre4 as F4
from fiber3 import Ops

LOOK = 3            # frozen for rung (4,4) on the T97 reviewer's advice (L = 2 breaks far from trained weights)


def _walk_down(model, st, B, nn, truth, w0, f_start, u_start, f_bound, u_bound, ops, cache, cons):
    """From u_start (clamp) downwards while misclassified; stop at the first correct point or at u_bound.
    Returns intervals [(lo, hi)] in increasing u (fn None = u 0)."""
    p = B @ w0
    T = st.L.shape[0]
    zero = np.zeros(3)
    fv = lambda f: zero if f is None else f
    f_u, u = f_start, u_start
    for _ in range(64):
        ug = st.gal @ w0; ops.add(3 * len(ug))
        d = st.gal - f_u
        same = st.ok & (np.abs(d).sum(1) < 1e-12 * (1 + np.abs(f_u).sum()))
        m = st.ok & ~same
        cons.append(d[m])
        ahead = m & (ug < u) & (ug > u_bound); ops.add(16 * len(ug))
        fb = fv(f_bound)
        if ahead.any():
            idx = np.flatnonzero(ahead); kk = int(idx[np.argmax(ug[idx])]); unext = float(ug[kk]); f_next = st.gal[kk]
            cons.append((st.gal[kk] - fb)[None])
            o = idx[idx != kk]
            cons.append(st.gal[o] - f_next); ops.add(4 * len(idx) + 4)
        else:
            kk = -1; unext = u_bound; f_next = f_bound
            o = np.flatnonzero(m & (ug < u))
            cons.append(st.gal[o] - fb); ops.add(4 * len(o))
        a = st.Lb @ w0; b = st.Ln; ops.add(3 * T)
        edges = []
        if st.live:
            for dcl in range(T):
                if dcl == truth:
                    continue
                den = b[dcl] - b[truth]
                if abs(den) < 1e-300:
                    cons.append((st.Lb[truth] - st.Lb[dcl])[None]); continue
                fd = (st.Lb[truth] - st.Lb[dcl]) / den; ops.add(4 + 16)
                cons.append((fd - f_u)[None]); cons.append((fd - fv(f_next))[None])
                v = float(fd @ w0)
                if unext < v < u:
                    edges.append((v, fd))
            if len(edges) > 1:
                for x in range(len(edges)):
                    for y in range(x + 1, len(edges)):
                        cons.append((edges[x][1] - edges[y][1])[None])
        edges.sort(key=lambda e: -e[0])
        seq = [(u, f_u)] + edges + [(unext, f_next)]
        stop = None
        for q in range(len(seq) - 1):
            vm = 0.5 * (seq[q][0] + seq[q + 1][0]); ops.add(2)
            cls = int(np.argmax(a + vm * b)) if st.live else 0; ops.add(2 * T)
            if cls == truth:
                stop = q; break
        if stop is not None:
            if stop > 0:
                fe = seq[stop][1]
                if kk >= 0:
                    cons.append((st.gal[kk] - fv(fe))[None])
                return [(fe, f_start)]
            if f_u is f_start:
                return []
            return [(f_u, f_start)]
        if kk < 0:
            return [(f_bound, f_start)]
        key = (id(st), kk, -1)
        if key not in cache:
            x = p + (unext - (abs(unext) * 1e-9 + 1e-12)) * nn; ops.add(12)
            cache[key] = F4.State(model, x, B, nn, ops)
        st = cache[key]; u = unext; f_u = f_next
    return [(f_u, f_start)]


def trace_F2L2(model, st0, B, nn, truth, cfn, w0, ops, cache):
    inf = {}
    iv, C = F4.trace(model, st0, B, nn, truth, cfn, w0, "F1", ops, cache, look=LOOK, info=inf)
    cons = [C] if len(C) else []
    cv = [float(c @ w0) for c in cfn]; ic = int(np.argmin(cv)); ucl = cv[ic]; fcl = cfn[ic]
    f_a = inf.get("f_stop"); u_a = inf.get("u_stop", 0.0)
    if ucl <= 0 or (f_a is not None and f_a is fcl) or (iv and iv[-1][1] is fcl):
        return iv, C
    x = B @ w0 + ucl * (1 - 1e-9) * nn; ops.add(12 + 16 + 4 * len(cfn))
    stc = F4.State(model, x, B, nn, ops)
    cache.setdefault("_alive", []).append(stc)   # keep stc alive: id(stc) is a cache key (T97 reviewer's fix)
    iv2 = _walk_down(model, stc, B, nn, truth, w0, fcl, ucl, f_a, u_a, ops, cache, cons)
    C2 = np.vstack([c if c.ndim == 2 else c[None] for c in cons if len(c)]) if cons else np.zeros((0, 3))
    return iv + iv2, C2


def face_err(model, fam, side, ops, stats):
    B = fam["B"]; nn, truth, clamps = side
    cfn = [F4._fn(w, B, nn) for w in clamps]; ops.add(16 * len(clamps))
    tot = 0.0
    for poly in fam["polys"]:
        for cell in F4.cells_on_face(model, poly, B, ops):
            w = np.mean(cell, 0); w /= np.linalg.norm(w)
            cache = {}
            st0 = F4.State(model, B @ w, B, nn, ops)
            stack = [cell]
            while stack:
                cur = stack.pop()
                w0 = np.mean(cur, 0); w0 /= np.linalg.norm(w0); ops.add(3 * len(cur) + 5)
                iv, C = trace_F2L2(model, st0, B, nn, truth, cfn, w0, ops, cache)
                stats["traces"] += 1
                if len(C):
                    s0 = C @ w0; ops.add(3 * len(C) + 4 * len(C))
                    keep = np.abs(C).sum(1) > 1e-13
                    for c, s in zip(C[keep], s0[keep]):
                        a_, b_ = F4.clip(cur, c if s > 0 else -c, ops)
                        if b_:
                            stack.append(b_)
                        cur = a_
                        if not cur:
                            break
                if not cur or not iv:
                    continue
                fns = []; sg = []
                for lo, hi in iv:
                    if hi is not None:
                        fns.append(hi); sg.append(1.0)
                    if lo is not None:
                        fns.append(lo); sg.append(-1.0)
                tot += F4.poly_int(cur, fns, sg, ops)
                stats["pieces"] += 1
    return tot


def est_F2L2_4(model, ops=None, info=None):
    ops = ops if ops is not None else Ops()
    stats = {"traces": 0, "pieces": 0}
    tot = 0.0
    for fam in F4.FAMS4:
        for side in fam["sides"]:
            tot += face_err(model, fam, side, ops, stats)
    if info is not None:
        info.update(stats)
    return 1.0 - tot / F4.S3

"""T73: mechanism-derived accuracy for any bias-free 2nd-argmax RNN with T = 3 inputs and any H ("arc trace").
Deductive (no random inputs); every coefficient is a formula of (Wi, Wh, Wo).

Mechanism used (README, rung M_6,3):
  * step 1: sign(x0) fixes the step-1 gates (chart).
  * step 2: neuron k switches on the plane A2_k . x = 0 (x1 = t_k x0); step 3 on A3_k . x = 0, where
    A3 = Wh diag(g2) A2 + Wi e2^T depends on the step-2 pattern. Readout L = Wo diag(g3) A3 x is linear per cell.
  * the network errs only in thin wedges along the true boundaries x_i = x_j (measured: diag_local.py, 99% of
    misclassified mass of the zoo M_6,3 lies within 0.4 deg of a boundary).
Estimator:
  1. Each ordering chamber is split between its two true boundaries by the equidistance plane (T63's 24
     model-independent half-chambers, `est_mech.HALVES`; not charged).
  2. For each half-chamber, walk its boundary arc and find where each gate switches ON THE ARC (step-2 roots,
     then step-3 roots inside each step-2 sub-arc with that sub-arc's A3). Only these gates are resolved: the
     half-chamber is cut by them, in arc order, into pieces that each own one arc segment of constant pattern.
  3. In each piece the pattern is frozen at its arc value (gates that do not cross the arc are not resolved);
     the logits are linear, and the error is the exact solid angle of the wrong-class slivers (T63's _err).
  4. tier "L" (lazy check): every wrong sliver is checked against the frozen gate rows; if a gate changes sign
     inside it, the piece is split by that gate and the far part re-read (recursively, depth <= MAXDEPTH). With
     this every counted wrong sliver is exactly wrong (no gate switches in it).
accuracy = 1 - sum of wrong masses.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
from m43 import Ops, clip, frac  # noqa: E402
from est_mech import HALVES, _err  # noqa: E402

E = np.eye(3)
MAXDEPTH = 6
TOL = 1e-12


def _arc_of(poly, n_true, tol=1e-9):
    on = [v for v in poly if abs(n_true @ v) < tol]
    return on if len(on) >= 2 else None


def _roots(rows, P, U, Th, ops):
    """theta in (0, Th) where rows . (cos t P + sin t U) = 0 (at most one per row since Th < pi)."""
    a = rows @ P; b = rows @ U; ops.add(6 * len(rows))
    out = []
    for k in range(len(rows)):
        # f(t) = a cos t + b sin t; root t0 = atan2(a, -b) mod pi
        t0 = np.arctan2(a[k], -b[k]) % np.pi; ops.add(2)
        if TOL < t0 < Th - TOL:
            out.append((t0, k))
    out.sort()
    return out


def _pt(P, U, t):
    return np.cos(t) * P + np.sin(t) * U


def _pattern_rows(model, sigma, x, ops):
    """Gates and gate rows at point x of chart sigma: (g2, g3, A2, A3, L)."""
    Wi, Wh, Wo = model
    H = len(Wi)
    g1 = ((sigma * Wi) > 0).astype(float)
    A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1]); ops.add(H * int(g1.sum()) + H)
    g2 = (A2 @ x > 0).astype(float); ops.add(3 * H)
    A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2]); ops.add(2 * H * int(g2.sum()) + H)
    g3 = (A3 @ x > 0).astype(float); ops.add(3 * H)
    L = Wo @ (g3[:, None] * A3); ops.add(3 * 3 * int(g3.sum()))
    return g2, g3, A2, A3, L


def _parts(L, g3, truth):
    if not g3.any():                      # all off: logits tie at 0, argmax -> class 0
        return [(0, [])]
    return [(truth, [L[truth] - L[o] for o in range(3) if o != truth])]


def _wrong_slivers(piece, L, g3, truth, ops):
    """Disjoint wrong-class slivers of piece under frozen linear logits (list of polygons)."""
    if not g3.any():
        return [] if truth == 0 else [piece]
    q = piece; out = []
    for o in range(3):
        if o == truth:
            continue
        ins, outp = clip(q, L[truth] - L[o], ops)
        if outp:
            out.append(outp)
        q = ins
        if not q:
            break
    return out


def _piece_err(model, sigma, piece, x_read, truth, ops, lazy, depth):
    g2, g3, A2, A3, L = _pattern_rows(model, sigma, x_read, ops)
    sl = _wrong_slivers(piece, L, g3, truth, ops)
    if not lazy or depth >= MAXDEPTH:
        return sum(frac(s, ops) for s in sl)
    rows = np.vstack([A2, A3]); ref = np.sign(rows @ x_read); ops.add(3 * len(rows))
    err = 0.0
    for s in sl:
        V = np.array(s); S = np.sign(V @ rows.T); ops.add(3 * len(rows) * len(s))
        bad = np.flatnonzero(((S * ref[None]) < 0).any(0))
        if len(bad) == 0:
            err += frac(s, ops); continue
        # a gate switches inside a wrong sliver: split the piece by it and redo both parts
        g = rows[bad[0]] * ref[bad[0]]                      # oriented so the reading point is inside
        near, far = clip(piece, g, ops)
        e = 0.0
        if near:
            e += _piece_err(model, sigma, near, x_read, truth, ops, lazy, depth + 1)
        if far:
            xf = np.mean(far, 0); xf /= np.linalg.norm(xf); ops.add(10)
            e += _piece_err(model, sigma, far, xf, truth, ops, lazy, depth + 1)
        return e
    return err


def half_err(model, h, ops, lazy=False):
    """Error mass inside one half-chamber h (dict from est_mech.HALVES)."""
    Wi, Wh, Wo = model
    sigma, truth, poly, n = h["sigma"], h["truth"], h["poly"], h["n_true"]
    on = _arc_of(poly, n)
    P, Q = on[0], on[-1]
    U = Q - (Q @ P) * P; U /= np.linalg.norm(U); Th = float(np.arccos(np.clip(P @ Q, -1, 1))); ops.add(12)
    H = len(Wi)
    g1 = ((sigma * Wi) > 0).astype(float)
    A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1]); ops.add(H * int(g1.sum()) + H)
    # step-2 crossings of the arc, then step-3 crossings inside each step-2 sub-arc
    cuts = []                                                   # (theta, plane row)
    r2 = _roots(A2, P, U, Th, ops)
    b2 = [0.0] + [t for t, _ in r2] + [Th]
    for s in range(len(b2) - 1):
        if s > 0:
            cuts.append((b2[s], A2[r2[s - 1][1]]))
        xm = _pt(P, U, 0.5 * (b2[s] + b2[s + 1])); ops.add(6)
        g2 = (A2 @ xm > 0).astype(float); ops.add(3 * H)
        A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2]); ops.add(2 * H * int(g2.sum()) + H)
        for t, k in _roots(A3, P, U, Th, ops):
            if b2[s] < t < b2[s + 1]:
                cuts.append((t, A3[k]))
    cuts.sort(key=lambda c: c[0])
    # cut the half-chamber in arc order; each piece owns the arc segment [t_prev, t]
    err = 0.0; cur = poly; tprev = 0.0
    for t, g in cuts + [(Th, None)]:
        if g is None:
            piece, rest = cur, []
        else:
            gg = g if g @ _pt(P, U, tprev + 0.5 * (t - tprev)) > 0 else -g; ops.add(9)
            piece, rest = clip(cur, gg, ops)
        if piece:
            xr = _pt(P, U, tprev + 0.5 * (t - tprev)); ops.add(6)
            err += _piece_err(model, sigma, piece, xr, truth, ops, lazy, 0)
        cur = rest; tprev = t
        if not cur:
            break
    return err


def est_arc3(model, ops=None, lazy=False, info=None):
    ops = ops if ops is not None else Ops()
    tot = 0.0; by = {}
    for h in HALVES:
        e = half_err(model, h, ops, lazy)
        tot += e
        if info is not None:
            i, j, k = h["circle"]
            key = (int(h["sigma"]), f"c{i}{j}{'+' if h['side'] == 1 else '-'}")
            by[key] = by.get(key, 0.0) + e
    if info is not None:
        info["err_by"] = by
    return 1.0 - tot

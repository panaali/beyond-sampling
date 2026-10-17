"""Exact accuracy of a (noisy) M_4,3 by linear-region enumeration (deductive; no random inputs).

The model has no biases, so every linear region is a polyhedral cone in R^3 and its Gaussian measure is a solid
angle / 4 pi. Start from the 12 simplicial cones {ordering of (x0,x1,x2)} x {sign x0}; split by the sign of each
hidden pre-activation at steps 1-3 (clipping convex spherical polygons, Sutherland-Hodgman on great circles); in
each final region the logits are linear, so the correct-answer region is two more halfspaces. Sum the solid angles
(fan triangulation, Van Oosterom-Strackee) of the correct pieces.

Cost is counted with m43.Ops (multiply-add = 1 op), every dot product, clip, matrix composition and solid angle.
"""
import numpy as np

from m43 import PERMS, Ops, clip, frac, order_normals, simplicial


def _split(poly, rows, ops):
    """All nonempty pieces of poly by the signs of rows.x; yields (piece, gate vector)."""
    pieces = [(poly, [])]
    for r in rows:
        nxt = []
        for p, g in pieces:
            ins, out = clip(p, r, ops)
            if ins:
                nxt.append((ins, g + [1.0]))
            if out:
                nxt.append((out, g + [0.0]))
        pieces = nxt
    return [(p, np.array(g)) for p, g in pieces]


def exact_acc(model, ops=None, stats=None):
    Wi, Wh, Wo = model
    ops = ops if ops is not None else Ops()
    e = np.eye(3)
    acc = 0.0; leaves = 0
    for perm in PERMS:
        c = perm[1]
        others = [o for o in range(3) if o != c]
        on = order_normals(perm)
        for s in (1.0, -1.0):
            poly = simplicial(np.array(on + [s * e[0]]))
            g1 = ((Wi * s) > 0).astype(float)                      # step-1 gates are fixed by sign(x0)
            A2 = Wh @ (g1[:, None] * np.outer(Wi, e[0])) + np.outer(Wi, e[1])
            ops.add(4 * int(g1.sum()) + 4)
            for p2, g2 in _split(poly, A2, ops):
                A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, e[2])
                ops.add(4 * int(g2.sum()) * 3 + 4)
                for p3, g3 in _split(p2, A3, ops):
                    if g3.sum() == 0:              # all h3 = 0: logits tie at 0, argmax returns index 0
                        if c == 0:
                            acc += frac(p3, ops); leaves += 1
                        continue
                    L = Wo @ (g3[:, None] * A3)
                    ops.add(3 * int(g3.sum()) * 3 + 6)
                    q = p3
                    for o in others:
                        q, _ = clip(q, L[c] - L[o], ops)
                        if not q:
                            break
                    if q:
                        acc += frac(q, ops); leaves += 1
    if stats is not None:
        stats["leaves"] = leaves
    return acc

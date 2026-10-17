"""Exact accuracy (and exact error mass per true-boundary family) of a bias-free 2nd-argmax RNN with T = 3, any H,
by linear-region enumeration (arc_m43_repro/exact.py, which is generic in H). Diagnostic / ground truth only."""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
from m43 import clip, frac  # noqa: E402
from exact import _split, exact_acc  # noqa: E402,F401
from est_mech import HALVES  # noqa: E402

E = np.eye(3)


def exact_by_family(model):
    """{(circle (i,j,k), s, truth): exact error mass} with s = +1 if x_k is above the pair. Same partition as
    fiber3.FAMS (half-chambers of both charts summed)."""
    Wi, Wh, Wo = model
    out = defaultdict(float)
    for hh in HALVES:
        sigma, circ, side, truth, poly = hh["sigma"], hh["circle"], hh["side"], hh["truth"], hh["poly"]
        key = (circ, side, truth)
        g1 = ((Wi * sigma) > 0).astype(float)
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
        for p2, g2 in _split(poly, A2, None):
            A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2])
            for p3, g3 in _split(p2, A3, None):
                tot = frac(p3)
                if g3.sum() == 0:
                    out[key] += 0.0 if truth == 0 else tot
                    continue
                L = Wo @ (g3[:, None] * A3)
                q = p3
                for o in range(3):
                    if o != truth:
                        q, _ = clip(q, L[truth] - L[o])
                        if not q:
                            break
                out[key] += tot - (frac(q) if q else 0.0)
    return dict(out)

"""Plain-language mechanism of a T = 3 zoo model, printed from closed-form quantities of the weights, and a check of
the predicted gate switches on the true boundary arcs against the actual network (forward passes at theta +- eps).
usage: python mechanism3.py H seed"""
import math
import sys

import numpy as np

from fiber3 import FAMS, Ops
from fibre import rows_at, trace_arc_rows
from zoo import zoo_model

np.set_printoptions(precision=2, suppress=True, linewidth=140)


def hidden_signs(model, x):
    Wi, Wh, Wo = model
    h = np.zeros(len(Wi)); out = []
    for t in range(len(x)):
        z = Wh @ h + Wi * x[t]; out.append(z > 0); h = np.maximum(z, 0)
    return np.concatenate(out)


def describe(model):
    Wi, Wh, Wo = model
    H = len(Wi); E = np.eye(3)
    print("Wi", Wi)
    for sg in (1, -1):
        g1 = sg * Wi > 0
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
        t = -A2[:, 0] / Wi
        print(f"chart sign(x0)={sg:+d}: step-1 neurons holding |x0|: {np.flatnonzero(g1).tolist()}; "
              f"step-2 neuron k switches at x1 = t_k x0, t = {t}")
    # pattern on each true boundary arc: which neurons are on at the final step, and how many segments
    tot_seg = 0; checks = 0; bad = 0
    for f in FAMS:
        segs = trace_arc_rows(model, f["P"], f["U"], 0.0, math.pi, Ops())
        tot_seg += len(segs)
        act = []
        for t0, t1, xm in segs:
            R, L, live = rows_at(model, xm, Ops())
            g3 = (R[2 * H:] @ xm) > 0
            act.append(int(g3.sum()))
            # check: the full pattern is constant inside the segment and changes at its ends (as predicted)
            if t1 - t0 < 1e-9:          # zero-length segment (x0 = 0 exactly): measure zero, skipped
                continue
            for th in (t0 + 1e-6 * (t1 - t0), t1 - 1e-6 * (t1 - t0)):
                p = math.cos(th) * f["P"] + math.sin(th) * f["U"]
                checks += 1
                bad += int(not np.array_equal(hidden_signs(model, p), hidden_signs(model, xm)))
        i, j, k = f["circle"]
        print(f"boundary x{i}=x{j}, x{k} {'above' if f['s'] > 0 else 'below'}: {len(segs)} constant-pattern arc segments; "
              f"final-step active neurons per segment: {act}")
    print(f"total arc segments {tot_seg}; predicted-switch check: {checks - bad}/{checks} segment ends have the "
          f"predicted pattern in the real network")


if __name__ == "__main__":
    describe(zoo_model(int(sys.argv[1]), 3, int(sys.argv[2])))

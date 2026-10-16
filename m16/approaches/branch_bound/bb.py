"""Best-first probabilistic branch and bound over (root, w-box) regions. Certified L <= acc <= U."""
import json
import sys
import time
import numpy as np
from core import (MS, TGT, root_boxes, measure, bounds, forward_x, FLOP_FWD, FLOP_BWD, FLOP_CROWNZ,
                  FLOP_MEAS)

RNN = 6000.0


def classify(root, wl, wu, model, inter):
    lb, ub, coef = bounds(root, wl, wu, model, inter)
    fin = np.isfinite(ub)
    correct = (lb > 0).all(1)
    wrong = ((ub < 0) & fin).any(1)
    # branching dim: worst margin's linear lower bound, |coef| * width
    jst = np.argmin(lb, 1)
    c = coef[np.arange(len(root)), jst]
    score = np.abs(c) * (wu - wl)
    return correct, wrong, score.argmax(1)


def center_correct(root, wl, wu, model):
    w = 0.5 * (wl + wu)
    x = np.einsum("ntd,nd->nt", MS[root], w)
    return forward_x(x, model).argmax(1) == TGT[root]


def run(model=None, inter="crown", batch=4096, max_flops=1e11, max_regions=3_000_000, log_every=1.6,
        verbose=True, max_time=1e9):
    fl_region = FLOP_FWD + FLOP_BWD + (FLOP_CROWNZ if inter == "crown" else 0) + FLOP_MEAS
    root = np.arange(90)
    wl, wu = root_boxes()
    mass = measure(wl, wu, 48)
    trunc = 1.0 - mass.sum()
    flops = 0.0
    Lm, Wm = 0.0, 0.0
    corr, wrong, sdim = classify(root, wl, wu, model, inter); flops += 90 * fl_region
    Lm += mass[corr].sum(); Wm += mass[wrong].sum()
    keep = ~(corr | wrong)
    root, wl, wu, mass, sdim = root[keep], wl[keep], wu[keep], mass[keep], sdim[keep]
    log, next_log, t0 = [], 90 * fl_region, time.time()
    while len(root) and flops < max_flops and len(root) < max_regions and time.time() - t0 < max_time:
        k = min(batch, len(root))
        sel = np.argpartition(-mass, k - 1)[:k] if k < len(root) else np.arange(len(root))
        rest = np.ones(len(root), bool); rest[sel] = False
        r, l, u, d = root[sel], wl[sel], wu[sel], sdim[sel]
        mid = 0.5 * (l[np.arange(k), d] + u[np.arange(k), d])
        l2, u2 = l.copy(), u.copy()
        u1 = u.copy(); u1[np.arange(k), d] = mid
        l2[np.arange(k), d] = mid
        cr = np.concatenate([r, r]); cl = np.concatenate([l, l2]); cu = np.concatenate([u1, u])
        cm = measure(cl, cu, 48 if (cu[:, 0] - cl[:, 0]).max() > 1 else 24)
        c, w, s = classify(cr, cl, cu, model, inter); flops += 2 * k * fl_region
        Lm += cm[c].sum(); Wm += cm[w].sum()
        kp = ~(c | w)
        root = np.concatenate([root[rest], cr[kp]]); wl = np.concatenate([wl[rest], cl[kp]])
        wu = np.concatenate([wu[rest], cu[kp]]); mass = np.concatenate([mass[rest], cm[kp]])
        sdim = np.concatenate([sdim[rest], s[kp]])
        if flops >= next_log:
            next_log = flops * log_every
            log.append(snapshot(flops, Lm, Wm, trunc, root, wl, wu, mass, model))
            if verbose:
                print(json.dumps(log[-1]), f"{time.time() - t0:.0f}s", flush=True)
    log.append(snapshot(flops, Lm, Wm, trunc, root, wl, wu, mass, model))
    if verbose:
        print(json.dumps(log[-1]), flush=True)
    return log, (root, wl, wu, mass)


def snapshot(flops, Lm, Wm, trunc, root, wl, wu, mass, model):
    cc = center_correct(root, wl, wu, model)
    und = 1.0 - Lm - Wm
    return dict(flops=flops, rnn_equiv=flops / RNN, n_open=int(len(root)), L=Lm, U=1.0 - Wm, gap=und,
                trunc=trunc, point_mid=Lm + 0.5 * und, point_center=Lm + mass[cc].sum() + 0.5 * trunc,
                eval_center_rnn=int(len(root)))


if __name__ == "__main__":
    inter = sys.argv[1] if len(sys.argv) > 1 else "crown"
    mf = float(sys.argv[2]) if len(sys.argv) > 2 else 1e11
    log, _ = run(inter=inter, max_flops=mf)
    json.dump(log, open(f"results_orig_{inter}.json", "w"), indent=1)

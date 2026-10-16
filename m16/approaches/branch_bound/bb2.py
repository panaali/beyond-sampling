"""Best-first probabilistic branch and bound on cone charts (cone.py). Certified L <= acc <= U."""
import json
import sys
import time
import numpy as np
import cone
from core import bounds, forward_x, FLOP_FWD, FLOP_BWD, FLOP_CROWNZ

RNN = 6000.0


def classify(root, wl, wu, model, inter):
    lb, ub, coef = bounds(root, wl, wu, model, inter, MS=cone.MS, TGT=cone.TGT)
    fin = np.isfinite(ub)
    correct = (lb > 0).all(1)
    wrong = ((ub < 0) & fin).any(1)
    ar = np.arange(len(root))
    jst = np.argmin(lb, 1)
    c = coef[ar, jst]
    score = np.abs(c) * (wu - wl)
    lo_, hi_ = lb[ar, jst], ub[ar, jst]
    pfrac = np.clip(hi_ / np.maximum(hi_ - lo_, 1e-12), 0, 1)        # crude "linear margin" share
    return correct, wrong, score.argmax(1), pfrac


def meas(root, wl, wu):
    wide = (wu[:, 0] - wl[:, 0]) > 0.25
    m = np.empty(len(root))
    if wide.any():
        m[wide] = cone.measure(cone.SGN[root[wide]], wl[wide], wu[wide], 24)
    if (~wide).any():
        m[~wide] = cone.measure(cone.SGN[root[~wide]], wl[~wide], wu[~wide], 12)
    return m


def center_eval(root, wl, wu, model):
    w = 0.5 * (wl + wu)
    base = np.maximum(wl, 0.5)
    geo = (wu > 4 * base); geo[:, 0] = False; geo[:, 9] = False
    w = np.where(geo, np.sqrt(base * wu), w)                           # representative point for wide eps ranges
    x = np.einsum("ntd,nd->nt", cone.MS[root], w)
    return forward_x(x, model).argmax(1) == cone.TGT[root]


def run(model=None, inter="crown", batch=4096, max_flops=1e11, max_regions=4_000_000, log_every=1.5,
        verbose=True, tag=""):
    fl_region = FLOP_FWD + FLOP_BWD + (FLOP_CROWNZ if inter == "crown" else 0) + cone.FLOP_MEAS
    root = np.arange(len(cone.ROOTS))
    wl, wu = cone.root_boxes()
    mass = meas(root, wl, wu)
    trunc = 1.0 - mass.sum()
    flops = len(root) * fl_region
    corr, wrong, sdim, pf = classify(root, wl, wu, model, inter)
    Lm, Wm = mass[corr].sum(), mass[wrong].sum()
    keep = ~(corr | wrong)
    root, wl, wu, mass, sdim, pf = root[keep], wl[keep], wu[keep], mass[keep], sdim[keep], pf[keep]
    log, next_log, t0 = [], flops, time.time()
    while len(root) and flops < max_flops and len(root) < max_regions:
        k = min(batch, len(root))
        sel = np.argpartition(-mass, k - 1)[:k] if k < len(root) else np.arange(len(root))
        rest = np.ones(len(root), bool); rest[sel] = False
        r, l, u, d = root[sel], wl[sel], wu[sel], sdim[sel]
        ar = np.arange(k)
        lo_, hi_ = l[ar, d], u[ar, d]
        base = np.maximum(lo_, 0.5)
        geo = (d >= 1) & (hi_ > 4 * base)                               # wide eps range: geometric split
        mid = np.where(geo, np.sqrt(base * hi_), 0.5 * (lo_ + hi_))
        u1 = u.copy(); u1[ar, d] = mid
        l2 = l.copy(); l2[ar, d] = mid
        cr = np.concatenate([r, r]); cl = np.concatenate([l, l2]); cu = np.concatenate([u1, u])
        m1 = meas(r, l, u1)
        pm = mass[sel]
        cm = np.concatenate([m1, np.maximum(pm - m1, 0.0)])            # children sum to parent exactly
        c, w, s, p = classify(cr, cl, cu, model, inter)
        flops += 2 * k * fl_region - k * cone.FLOP_MEAS
        Lm += cm[c].sum(); Wm += cm[w].sum()
        kp = ~(c | w)
        root = np.concatenate([root[rest], cr[kp]]); wl = np.concatenate([wl[rest], cl[kp]])
        wu = np.concatenate([wu[rest], cu[kp]]); mass = np.concatenate([mass[rest], cm[kp]])
        sdim = np.concatenate([sdim[rest], s[kp]]); pf = np.concatenate([pf[rest], p[kp]])
        if flops >= next_log:
            next_log = flops * log_every
            log.append(snapshot(flops, Lm, Wm, trunc, root, wl, wu, mass, model, pf))
            if verbose:
                print(tag, json.dumps(log[-1]), f"{time.time() - t0:.0f}s", flush=True)
    log.append(snapshot(flops, Lm, Wm, trunc, root, wl, wu, mass, model, pf))
    if verbose:
        print(tag, json.dumps(log[-1]), flush=True)
    return log, (root, wl, wu, mass, pf)


def snapshot(flops, Lm, Wm, trunc, root, wl, wu, mass, model, pf):
    cc = center_eval(root, wl, wu, model)
    und = 1.0 - Lm - Wm
    return dict(flops=flops, rnn_equiv=flops / RNN, n_open=int(len(root)), L=float(Lm), U=float(1.0 - Wm),
                gap=float(und), trunc=float(trunc), point_mid=float(Lm + 0.5 * und),
                point_center=float(Lm + mass[cc].sum() + 0.5 * trunc), point_lin=float(Lm + (mass * pf).sum() + 0.5 * trunc),
                center_evals=int(len(root)))


if __name__ == "__main__":
    inter = sys.argv[1] if len(sys.argv) > 1 else "crown"
    mf = float(sys.argv[2]) if len(sys.argv) > 2 else 1e11
    log, _ = run(inter=inter, max_flops=mf, log_every=1.25)
    json.dump(log, open(f"results_cone_orig_{inter}_{mf:.0e}.json", "w"), indent=1)

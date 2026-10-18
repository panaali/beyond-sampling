"""T317c: T317b (cont.est, G = 20) with two exact, executed cost-only savings:
(1) touching correct pieces along a line (cell k ends where cell k+1 starts, same float) are merged before the Phi
    profile, so shared inner endpoints are not evaluated (telescoping; exact up to rounding);
(2) endpoints at exactly 0 use the model-free table Phi(v2 nodes) (precomputed, uncharged); pieces reaching tmax merge
    with the model-free tail [tmax, inf).
correct_interval is charged at the scorer's 216 FLOPs (was 146). Estimates must equal T317b to <= 1e-12."""
import os
import sys

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont  # noqa: E402
from cont import r0, W, r1w, r1w2, H, T, C, F_STEP, F_TAN, F_ROOT  # noqa: E402

DEDUCTIVE = True
F_CI = 229                                    # correct_interval (official count, T281 scorer 379498b6)
F_CELL = 2 * 2 * C * H + F_CI + 4             # first-cell readout + correct set + bookkeeping
F_LATER_READ = 2 * C + 2 * C * H + F_CI + 4
F_PHI = 20
TMAX = r1w2.TMAX


def walk_counted(model, x0, d, k, tmax):
    """cont.walk_cont with the honest correct_interval count (same executed arithmetic; FLOPs recounted)."""
    pieces, f_old = cont.walk_cont(model, x0, d, k, tmax)
    n_first = len(x0); n_later = sum(len(p[0]) for p in pieces) - n_first
    f = f_old + n_first * (F_CELL - cont.F_CELL) + n_later * (F_LATER_READ - cont.F_LATER_READ)
    return pieces, f


def merge(pieces, n, tmax):
    """Per row, merge touching non-empty correct pieces in walk order; a piece ending at tmax merges with the model-free
    tail. Returns (row, a, b) arrays with b = INF for merged tails, and the comparison count."""
    cur_a = np.full(n, np.nan); cur_b = np.full(n, np.nan); out_r, out_a, out_b = [], [], []; ops = 0
    for rows, lo, hi in pieces:
        m = hi > lo; r, a, b = rows[m], lo[m], hi[m]; ops += 2 * len(rows)
        touch = cur_b[r] == a
        cur_b[r[touch]] = b[touch]
        new = ~touch
        old = r[new][~np.isnan(cur_a[r[new]])]
        out_r.append(old); out_a.append(cur_a[old]); out_b.append(cur_b[old])
        cur_a[r[new]] = a[new]; cur_b[r[new]] = b[new]
    open_ = np.flatnonzero(~np.isnan(cur_a))
    out_r.append(open_); out_a.append(cur_a[open_]); out_b.append(cur_b[open_])
    R = np.concatenate(out_r); A = np.concatenate(out_a); B = np.concatenate(out_b)
    reach = B >= tmax                                     # merges with the tail [tmax, inf)
    B = np.where(reach, W.INF, B)
    tail_rows = np.setdiff1d(np.arange(n), R[reach])      # rows whose last piece does not reach tmax keep the tail alone
    R = np.concatenate([R, tail_rows]); A = np.concatenate([A, np.full(len(tail_rows), tmax)])
    B = np.concatenate([B, np.full(len(tail_rows), W.INF)])
    return R, A, B, ops


def profile(R, A, B, n, nodes, kind, tmax, phi0):
    """Sum over merged intervals; Phi at model-free arguments (a = 0 -> phi0 table; b = inf; a = tmax) is not charged."""
    out = np.zeros((n, len(nodes))); f = 0.0
    a0 = A == 0.0; binf = B >= W.INF - 1e-12; atm = A == tmax
    # evaluate Phi only where the argument is weight-dependent (masked BEFORE the ndtr call: savings as executed)
    Pa = np.empty((len(R), len(nodes))); Pb = np.empty((len(R), len(nodes)))
    ia = np.flatnonzero(~a0); ib = np.flatnonzero(~binf)
    if kind == "12":
        Pa[a0] = phi0["p"][None]; Pa[ia] = ndtr(nodes[None] + A[ia, None])
        Pb[binf] = 1.0; Pb[ib] = ndtr(nodes[None] + B[ib, None])
        val = Pb - Pa
    else:
        Pa[a0] = phi0["m8"][None]; Pa[ia] = ndtr(nodes[None] - A[ia, None]) ** 8
        Pb[binf] = 0.0; Pb[ib] = ndtr(nodes[None] - B[ib, None]) ** 8
        val = Pa - Pb
    np.add.at(out, R, val)
    nphi = len(ia) + len(ib)              # Phi calls actually made (a = tmax tail starts are evaluated, so charged)
    f += len(nodes) * (nphi * (F_PHI + (3 if kind == "23" else 0)) + 2 * len(R))
    return out, f


def est(model, budget=None, G=20):
    model = tuple(np.asarray(w, np.float64) for w in model)
    TRI = r1w.TRI; NT = len(TRI); k = TRI[:, 1]; flops = 0.0; tot = np.zeros(NT)
    for name, lo, hi, (v2s, g12s, g23s) in r1w2.REG[False]:
        nodes, wts = cont._gl(lo, hi, G)
        phi0 = dict(p=ndtr(nodes), m8=ndtr(nodes) ** 8)               # model-free tables
        v3s = v2s - g23s; v4s = float(r1w.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
        m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
        x, dx = r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)
        p23, f = walk_counted(model, x, dx, k, TMAX); flops += f
        R, A, B, ops = merge(p23, NT, TMAX); flops += ops
        I23, f = profile(R, A, B, NT, nodes, "23", TMAX, phi0); flops += f
        if name.startswith("nb"):
            x, dx = r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0)
            p12, f = walk_counted(model, x, dx, k, TMAX); flops += f
            R, A, B, ops = merge(p12, NT, TMAX); flops += ops
            I12, f = profile(R, A, B, NT, nodes, "12", TMAX, phi0); flops += f
        else:
            I12 = np.repeat((1.0 - phi0["p"])[None], NT, 0)                # blind: model-free (V1)
        tot += (I12 * I23 * (90 * norm.pdf(nodes))[None]) @ wts
        flops += NT * len(nodes) * 4
    flops += NT
    return float(tot.mean()), float(flops)


def honest_baseline_cost(model, G=20):
    """T317b's cost with correct_interval at 216 (no new savings): the reference for the >= 3% test."""
    a, f = cont.est(model, G=G)
    # recount: cont charged F_CELL (146-based) and F_LATER_READ (146-based) per cell; add the 70-FLOP difference per cell
    model = tuple(np.asarray(w, np.float64) for w in model); k = r1w.TRI[:, 1]; ncells = 0
    for name, lo, hi, (v2s, g12s, g23s) in r1w2.REG[False]:
        v3s = v2s - g23s; v4s = float(r1w.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
        m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
        lines = [r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)]
        if name.startswith("nb"):
            lines.append(r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0))
        for x, dx in lines:
            p, _ = cont.walk_cont(model, x, dx, k, TMAX); ncells += sum(len(q[0]) for q in p)
    return a, f + ncells * (F_CI - 146)

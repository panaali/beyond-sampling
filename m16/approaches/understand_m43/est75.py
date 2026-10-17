"""T75: generic boundary expansion of order K for M_4,3-type models (any trained seed). Deductive, no random inputs.

Recipe (no per-seed hand account; nothing fitted to model outputs):
  * Families (fixed geometry, from est_mech.HALVES): each ordering chamber x chart sign(x0) is split between its two
    adjacent true boundaries x_i = x_j by the equidistance plane -> 24 half-chambers, each with a known true class and
    an edge on its true boundary plane (the "arc").
  * Step by step (t = 2, 3), inside each current cell: a gate (row a of the step's pre-activation map, a closed-form
    matrix product of the weights restricted to the cell's earlier gate pattern) that does not cut the cell has a known
    state (no approximation). Among the gates that cut the cell, the K_t closest to the boundary arc (angular distance
    of the arc to the gate plane; 0 if the gate crosses the arc) are RESOLVED (the cell is split by them); the others
    are FROZEN at their state on the arc (first order in the distance from the boundary).
  * In each final cell the logits are linear, l = Wo diag(g3) A3 x; the error is the mass cut off while clipping the
    cell by the two "true class wins" halfspaces (all-off cell: argmax tie -> class 0).
  K = (4, 4) resolves every cutting gate and is exact; smaller K is the expansion. Cost per half-chamber is at most
  prod_t 2^K_t cells, independent of how many linear regions the model has elsewhere.

Ops: strict count (review/understand_m43/ops_audit.py convention): clip 4 per vertex + 13 per new vertex, solid-angle
triangle 23 (+1), every comparison / sign test 1, arc-point search 4 per vertex, matrix products by multiply-adds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
from m43 import Ops  # noqa: E402
from est_mech import HALVES  # noqa: E402  (model-independent geometry, not charged)

E = np.eye(3)


# ------------------------------------------------------------------ strict-count geometry (own copies, same maths)
DEBUG = None
TIE = 0.0          # 1e-9 for T85b (set by est75(cliptol>0)); 0 keeps T85 as evaluated
CLIP_TOL = [0.0]     # relative tolerance for "vertex on the plane"; 0 for T75/T85 (as evaluated), 1e-12 for T85b


def clip(poly, n, ops):
    s = [float(n @ v) for v in poly]
    ops.add(4 * len(poly))
    if CLIP_TOL[0] > 0:
        tl = CLIP_TOL[0] * max(abs(x) for x in s)
        s = [0.0 if abs(x) <= tl else x for x in s]
    if min(s) >= 0:
        return poly, []
    if max(s) <= 0:
        return [], poly
    ins, out = [], []
    m = len(poly)
    for i in range(m):
        j = (i + 1) % m
        vi, si, sj = poly[i], s[i], s[j]
        if si >= 0:
            ins.append(vi)
        if si <= 0:
            out.append(vi)
        if (si > 0 and sj < 0) or (si < 0 and sj > 0):
            p = abs(si) * poly[j] + abs(sj) * vi
            p = p / np.linalg.norm(p); ops.add(13)
            ins.append(p); out.append(p)
    return (ins if len(ins) >= 3 else []), (out if len(out) >= 3 else [])


def frac(poly, ops):
    if len(poly) < 3:
        return 0.0
    a = poly[0]; tot = 0.0
    for i in range(1, len(poly) - 1):
        b, c = poly[i], poly[i + 1]
        tot += 2.0 * np.arctan2(abs(a @ np.cross(b, c)), 1.0 + a @ b + a @ c + b @ c)
        ops.add(23)
    ops.add(1)
    return tot / (4 * np.pi)


def _arc(poly, n_true, ops, tol=1e-9):
    ops.add(4 * len(poly))
    return [v for v in poly if abs(n_true @ v) < tol]


STATS = {"cut": 0, "resolved": 0, "cells": 0}


def _resolve(cells, rows, n_true, K, ops, rule="arc", w=None, eps=0.1):
    """Split each cell by the cutting gates that are resolved; freeze the others at their state at the cell's arc
    midpoint (vertex mean if the cell has no arc edge). Rules for choosing what to resolve:
      'arc'      up to K gates, closest to the arc first (T63 tier C style);
      'impact'   up to K gates, largest impact first, impact_k = depth_k * w_k with depth_k the largest
                 pre-activation reached on the side opposite to the frozen state and w_k the neuron's largest
                 downstream weight difference;
      'adaptive' every cutting gate whose impact is >= eps * (sum of impacts of the neurons that are on in the whole
                 cell), i.e. freeze a neuron only if it cannot move the readout by more than a fraction eps of what
                 the surely-on neurons contribute; if no neuron is surely on, resolve all cutting gates. Cap K."""
    out = []
    for poly in cells:
        on = _arc(poly, n_true, ops)
        ref = np.mean(on, 0) if len(on) >= 2 else np.mean(poly, 0); ops.add(3 * len(poly))
        g = np.zeros(len(rows)); cut = []; scale = 0.0
        for k, a in enumerate(rows):
            s = [a @ v for v in poly]; ops.add(4 * len(poly))
            if min(s) >= 0:
                g[k] = 1.0
                if w is not None:
                    scale += max(s) * w[k]; ops.add(1 + len(s))
            elif max(s) <= 0:
                g[k] = 0.0
            else:
                g[k] = float(a @ ref > 0); ops.add(4)
                if rule == "arc":
                    if len(on) >= 2:
                        so = [a @ v for v in on]; ops.add(4 * len(on))
                        d = 0.0 if min(so) < 0 < max(so) else min(abs(x) for x in so) / np.linalg.norm(a)
                        ops.add(5)
                    else:
                        d = np.inf
                    cut.append((d, k))
                else:
                    depth = max(s) if g[k] == 0 else -min(s); ops.add(len(s))
                    cut.append((-depth * w[k], k)); ops.add(1)
        cut.sort(); ops.add(len(cut) * 2)
        if rule == "adaptive":
            chosen = [k for d, k in cut if -d >= eps * scale][:K]; ops.add(2 * len(cut))
        else:
            chosen = [k for d, k in cut[:K]]
        STATS["cut"] += len(cut); STATS["resolved"] += len(chosen)
        pieces = [(poly, g)]
        for k in chosen:
            nxt = []
            for p, gg in pieces:
                ins, outp = clip(p, rows[k], ops)
                if ins:
                    g1 = gg.copy(); g1[k] = 1.0; nxt.append((ins, g1))
                if outp:
                    g0 = gg.copy(); g0[k] = 0.0; nxt.append((outp, g0))
            pieces = nxt
        out += pieces
    return out


def _margin(L, truth, V):
    """m(v) = l_truth(v) - max_other l(v) at the points V (rows of an array)."""
    Z = V @ L.T
    oth = np.max(np.delete(Z, truth, axis=1), axis=1)
    return Z[:, truth] - oth


def _cells_margin(poly, A3, Wo, truth, n_true, ops, eps, done=()):
    """T85 margin-aware step-3 rule. Freeze every cutting gate at its arc state, then resolve (split by) a cutting
    gate k only if, on its minority side (the part of the cell where its true state differs from the frozen one),
    the logit shift it can cause, |a_k . v| * w_k with w_k = max_o |Wo[truth,k] - Wo[o,k]|, reaches the frozen
    readout margin m(v) = l_truth - max_other l at some point, or the margin changes sign there with a shift of at
    least eps * max|m|. Points tested: minority vertices and the points where the gate plane cuts the cell's edges
    (m - shift is concave on a cell, so vertex tests bound the whole region). Recursive: children are re-tested."""
    on = _arc(poly, n_true, ops)
    ref = np.mean(on, 0) if len(on) >= 2 else np.mean(poly, 0); ops.add(3 * len(poly))
    V = np.array(poly)
    S = V @ A3.T; ops.add(4 * len(poly) * len(A3))
    g = np.zeros(len(A3)); cut = []
    tol = 1e-12 * np.abs(S).max(0)
    for k in range(len(A3)):
        if S[:, k].min() >= -tol[k]:
            g[k] = 1.0 if S[:, k].max() > tol[k] else float(A3[k] @ ref > 0)
        elif S[:, k].max() <= tol[k]:
            g[k] = 0.0
        else:
            g[k] = float(A3[k] @ ref > 0); ops.add(4)
            if k not in done:
                cut.append(k)
    if not done:
        STATS["cut"] += len(cut)
    if not cut:
        return [(poly, g)]
    w = np.max(np.abs(Wo[truth][None, :] - Wo), 0); ops.add(3 * len(A3))
    L = Wo @ (g[:, None] * A3); ops.add(3 * int(g.sum()) * 3)
    trig = []
    for k in cut:
        sk = S[:, k]
        minority = (sk > 0) != (g[k] > 0)
        pts = [V[q] for q in np.flatnonzero(minority)]
        shifts = [abs(sk[q]) * w[k] for q in np.flatnonzero(minority)]
        m_ = len(poly)
        for q in range(m_):                         # gate-plane / edge crossing points (shift 0 there)
            r = (q + 1) % m_
            if (sk[q] > 0) != (sk[r] > 0) and sk[q] != 0 and sk[r] != 0:
                pp = abs(sk[q]) * V[r] + abs(sk[r]) * V[q]; pts.append(pp / np.linalg.norm(pp)); shifts.append(0.0); ops.add(13)
        P = np.array(pts); sh = np.array(shifts)
        if g.sum() == 0:
            mm = np.full(len(P), 1.0 if truth == 0 else -1.0) * 0.0
        else:
            mm = _margin(L, truth, P); ops.add(len(P) * (3 * 3 + 3))
        score = np.max(sh / (np.abs(mm) + 1e-300)); ops.add(2 * len(P))
        flips = mm.min() < 0 < mm.max() and sh.max() >= eps * np.abs(mm).max()
        if DEBUG is not None:
            DEBUG.append((tuple(done), len(poly), k, float(score), bool(flips), float(sh.max()), float(np.abs(mm).max()), float(mm.min()), float(mm.max())))
        if score >= 1 - TIE or flips:      # ties (shift == margin exactly) are structural: count them as triggering
            trig.append(k)
    if not trig:
        return [(poly, g)]
    # resolve every triggering gate at once (order-independent), then re-test the children
    STATS["resolved"] += len(trig)
    pieces = [poly]
    for k in trig:
        nxt = []
        for q in pieces:
            ins, outp = clip(q, A3[k], ops)
            nxt += [x for x in (ins, outp) if x]
        pieces = nxt
    out = []
    for q in pieces:
        out += _cells_margin(q, A3, Wo, truth, n_true, ops, eps, tuple(done) + tuple(trig))
    return out


def _resolve_margin2(poly, A2, model, truth, n_true, ops, eps, done=()):
    """T112 certified-style step-2 rule (intermediate step). Freeze every cutting step-2 gate at its arc state; resolve
    gate k only if on its minority side the change it can cause in the readout margin, |a_k . v| * w_k with the
    Lipschitz bound w_k = max_o sum_j |(Wo[t,j] - Wo[o,j]) Wh[j,k]| (ReLU is 1-Lipschitz), reaches the margin of the
    frozen network m(v) (full forward of the frozen model at v), or that margin changes sign there with a change
    >= eps * max|m|. Test points: minority vertices and gate/edge crossing points. All triggering gates are split at
    once and the children re-tested (order-independent). Scale-invariant like _resolve_inv."""
    Wi, Wh, Wo = model
    on = _arc(poly, n_true, ops)
    ref = np.mean(on, 0) if len(on) >= 2 else np.mean(poly, 0); ops.add(3 * len(poly))
    V = np.array(poly); S = V @ A2.T; ops.add(4 * len(poly) * len(A2))
    tol = 1e-12 * np.abs(S).max(0)
    g = np.zeros(len(A2)); cut = []
    for k in range(len(A2)):
        if S[:, k].min() >= -tol[k]:
            g[k] = 1.0 if S[:, k].max() > tol[k] else float(A2[k] @ ref > 0)
        elif S[:, k].max() <= tol[k]:
            g[k] = 0.0
        else:
            g[k] = float(A2[k] @ ref > 0); ops.add(4)
            if k not in done:
                cut.append(k)
    if not done:
        STATS["cut"] += len(cut)
    if not cut:
        return [(poly, g)]
    D = np.abs(Wo[truth][None, :] - Wo)
    w = np.max(D @ np.abs(Wh), 0); ops.add(3 * len(A2) * len(A2) + 3 * len(A2))
    A3f = Wh @ (g[:, None] * A2) + np.outer(Wi, E[2]); ops.add(len(A2) * int(g.sum()) * 2)

    def margin(P):
        Z = np.maximum(P @ A3f.T, 0) @ Wo.T; ops.add(len(P) * (4 * len(A2) + 3 * len(A2)))
        if not np.any(np.maximum(P @ A3f.T, 0)):
            return np.zeros(len(P))
        return Z[:, truth] - np.max(np.delete(Z, truth, axis=1), axis=1)

    trig = []
    for k in cut:
        sk = S[:, k]
        mino = (sk > 0) != (g[k] > 0)
        pts = [V[q] for q in np.flatnonzero(mino)]; sh = [abs(sk[q]) * w[k] for q in np.flatnonzero(mino)]
        m_ = len(poly)
        for q in range(m_):
            r = (q + 1) % m_
            if (sk[q] > 0) != (sk[r] > 0) and sk[q] != 0 and sk[r] != 0:
                pp = abs(sk[q]) * V[r] + abs(sk[r]) * V[q]; pts.append(pp / np.linalg.norm(pp)); sh.append(0.0); ops.add(13)
        P = np.array(pts); sh = np.array(sh); mm = margin(P)
        score = np.max(sh / (np.abs(mm) + 1e-300)); ops.add(2 * len(P))
        flips = mm.min() < 0 < mm.max() and sh.max() >= eps * np.abs(mm).max()
        if score >= 1 - TIE or flips:
            trig.append(k)
    if not trig:
        return [(poly, g)]
    STATS["resolved"] += len(trig)
    pieces = [poly]
    for k in trig:
        nxt = []
        for q in pieces:
            ins, outp = clip(q, A2[k], ops)
            nxt += [x for x in (ins, outp) if x]
        pieces = nxt
    out = []
    for q in pieces:
        out += _resolve_margin2(q, A2, model, truth, n_true, ops, eps, tuple(done) + tuple(trig))
    return out


def _resolve_inv(poly, A2, model, truth, n_true, ops, eps, bound=False):
    """T85b scale-invariant step-2 rule. Like 'adaptive', but the weight factor of step-2 neuron k is its effect on
    the readout margin through the step-3 neurons that are on at the cell's arc point: v_k = Wo diag(g3_ref) Wh[:, k],
    w_k = max_o |v_k[truth] - v_k[o]|. Rescaling a neuron by c > 0 (input row x c, output column / c) leaves
    depth_k * w_k unchanged, so the decision is a property of the function, not of the parametrisation."""
    Wi, Wh, Wo = model
    on = _arc(poly, n_true, ops)
    ref = np.mean(on, 0) if len(on) >= 2 else np.mean(poly, 0); ops.add(3 * len(poly))
    V = np.array(poly); S = V @ A2.T; ops.add(4 * len(poly) * len(A2))
    g = np.zeros(len(A2)); cut = []
    gref = (A2 @ ref > 0).astype(float); ops.add(4 * len(A2))
    A3r = Wh @ (gref[:, None] * A2) + np.outer(Wi, E[2]); ops.add(len(A2) * int(gref.sum()) * 2)
    g3r = (A3r @ ref > 0).astype(float); ops.add(4 * len(A2))
    if bound:
        # triangle-inequality bound over ALL step-3 neurons: w_k = max_o sum_j |(Wo[t,j] - Wo[o,j]) Wh[j,k]|
        D = np.abs(Wo[truth][None, :] - Wo)                       # (classes, step-3 neurons)
        w = np.max(D @ np.abs(Wh), 0); ops.add(3 * len(A2) * len(A2) + 3 * len(A2))
    else:
        Veff = Wo @ (g3r[:, None] * Wh); ops.add(3 * int(g3r.sum()) * len(A2))      # (classes, step-2 neurons)
        w = np.max(np.abs(Veff[truth][None, :] - Veff), 0); ops.add(3 * len(A2))
    scale = 0.0
    tol = 1e-12 * np.abs(S).max(0)
    for k in range(len(A2)):
        if S[:, k].min() >= -tol[k]:
            g[k] = 1.0 if S[:, k].max() > tol[k] else gref[k]
            scale += S[:, k].max() * w[k] * g[k]; ops.add(1 + len(poly))
        elif S[:, k].max() <= tol[k]:
            g[k] = 0.0
        else:
            g[k] = gref[k]
            depth = S[:, k].max() if g[k] == 0 else -S[:, k].min(); ops.add(len(poly) + 1)
            cut.append((depth * w[k], k))
    cut.sort(reverse=True); ops.add(2 * len(cut))
    chosen = [k for imp, k in cut if imp >= eps * scale] if scale > 0 else [k for imp, k in cut]
    STATS["cut"] += len(cut); STATS["resolved"] += len(chosen)
    pieces = [(poly, g)]
    for k in chosen:
        nxt = []
        for p, gg in pieces:
            ins, outp = clip(p, A2[k], ops)
            if ins:
                g1 = gg.copy(); g1[k] = 1.0; nxt.append((ins, g1))
            if outp:
                g0 = gg.copy(); g0[k] = 0.0; nxt.append((outp, g0))
        pieces = nxt
    return pieces


def est75(model, ops=None, K=(4, 2), info=None, rule="arc", eps=0.1, cliptol=0.0):
    global TIE
    CLIP_TOL[0] = cliptol; TIE = 1e-9 if cliptol > 0 else 0.0
    try:
        return _est75(model, ops, K, info, rule, eps)
    finally:
        CLIP_TOL[0] = 0.0; TIE = 0.0


def _est75(model, ops, K, info, rule, eps):
    Wi, Wh, Wo = model
    ops = ops if ops is not None else Ops()
    err_by = {}
    for h in HALVES:
        sigma, truth, n_true = h["sigma"], h["truth"], h["n_true"]
        i, j, k = h["circle"]
        key = (int(sigma), f"c{i}{j}{'+' if h['side'] == 1 else '-'}")
        g1 = ((sigma * Wi) > 0).astype(float); ops.add(4)
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1]); ops.add(4 * int(g1.sum()))
        err = 0.0
        w2 = np.abs(Wh).max(0) if rule != "arc" else None      # step-2 neuron -> largest effect on step-3 rows
        w3 = np.max(np.abs(Wo[truth][None, :] - Wo), 0) if rule != "arc" else None
        if rule != "arc":
            ops.add(16 + 12)
        r2 = "adaptive" if rule == "margin" else rule
        e2 = eps[0] if isinstance(eps, tuple) else eps
        cells2 = _resolve_margin2(h["poly"], A2, model, truth, n_true, ops, e2) if rule == "margin_all" else \
            _resolve_inv(h["poly"], A2, model, truth, n_true, ops, e2, rule == "margin_inv2") if rule in ("margin_inv", "margin_inv2") else \
            _resolve([h["poly"]], A2, n_true, K[0], ops, r2, w2, e2)
        for p2, g2 in cells2:
            A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2]); ops.add(4 * int(g2.sum()) * 2)
            e3 = eps[1] if isinstance(eps, tuple) else eps
            cells3 = _cells_margin(p2, A3, Wo, truth, n_true, ops, e3) if rule in ("margin", "margin_inv", "margin_inv2", "margin_all") else \
                _resolve([p2], A3, n_true, K[1], ops, rule, w3, e3)
            for p3, g3 in cells3:
                STATS["cells"] += 1
                if g3.sum() == 0:
                    if truth != 0:
                        err += frac(p3, ops)
                    continue
                L = Wo @ (g3[:, None] * A3); ops.add(3 * int(g3.sum()) * 3)
                q = p3
                for o in range(3):
                    if o == truth:
                        continue
                    ins, outp = clip(q, L[truth] - L[o], ops); ops.add(3)
                    if outp:
                        err += frac(outp, ops)
                    q = ins
                    if not q:
                        break
        err_by[key] = err_by.get(key, 0.0) + err
    if info is not None:
        info["err_by"] = err_by
    return 1.0 - sum(err_by.values())


# The frozen T75 recipe (chosen on dev noise seeds 0 of zoo seeds 0-4 before the fresh evaluation)
RECIPE = dict(K=(4, 4), rule="adaptive", eps=0.1)

# T85 recipe at rung (H=4, T=3): step 2 as T75 (adaptive impact), step 3 margin-aware (frozen before fresh seeds 40-43)
RECIPE85 = dict(K=(4, 4), rule="margin", eps=0.1)

# T85b: scale-invariant step-2 impact + margin-aware step 3 (frozen before fresh noise seeds 60-63)
RECIPE85B = dict(K=(4, 4), rule="margin_inv2", eps=0.1, cliptol=1e-12)

# T112: margin certificate at BOTH steps (intermediate step 2 via the Lipschitz readout bound, step 3 as T85);
# chosen after the clean-seed diagnosis at H = 4, 5, 6, 8 (T = 3), frozen before fresh noise seeds 70-73.
RECIPE112 = dict(K=(99, 99), rule="margin_all", eps=0.1, cliptol=1e-12)

"""T63: accuracy of M_4,3 derived from its mechanism ("pair-ratio boundary account"). Deductive: no random inputs.

Mechanism (algzoo/labels_4_3.py; every coefficient below is computed from (Wi, Wh, Wo), nothing is fitted):
  * step 1: sign(x0) picks which mirror half of the hidden layer carries x0 (chart sigma = sign x0).
  * step 2: the step-2 gates switch on the four planes x1 = t_k x0, t_k = -A2_k0 / Wi_k, A2 = Wh diag(g1) Wi e0^T
    + Wi e1^T. They cut each chart into 5 strips in r = x1/x0; the middle strip, containing x1 = x0, is the soft
    "x1 vs x0" switch of the labels.
  * step 3: each neuron is h_k = ReLU(A3_k . x), A3 = Wh diag(g2) A2 + Wi e2^T (closed form per strip), i.e.
    h_k = |Wi_k| (+-(x2 - tau_k(x0, x1)))_+ with a threshold tau_k that is linear in each strip.
  * readout: near a true decision boundary only TWO neurons (a, b) are on, so l_i = Wo[i,a] h_a + Wo[i,b] h_b and the
    class depends only on the ratio nu = h_b / h_a. The upper envelope of the three lines Wo[i,a] + Wo[i,b] nu gives
    thresholds nu_1 < nu_2 and the class order in closed form. Both off ("dead") gives the argmax tie -> class 0.
    The pair is the u pair {0,1} when x2 is above the two compared inputs, the v pair {2,3} when below, and a
    crossover pair {outer of one mirror half, inner of the other} on the boundary x1 = x2; it is read off the
    weights at one anchor point near the x0 -> 0 end of each boundary arc.

Error families: for each true boundary circle x_i = x_j, side (third input k max or min), chart sigma, truth side and
step-2 strip, the error is the mass of the "wrong-class" ratio sectors inside the half-chamber assigned to this
boundary (each ordering chamber is split between its two boundaries by the plane on which the median is equidistant
from its two neighbours, so families are disjoint). Accuracy = 1 - sum of these cone masses (exact solid angles of
convex spherical polygons, Van Oosterom-Strackee = Girard).

Refinement for the crossover boundary x1 = x2 ("resolve"): the step-3 gate planes that cross the true boundary arc
inside a strip piece are resolved (the piece is split by them) and the active set is read on the arc in each
sub-piece; gates that do not cross the arc are frozen at their value on the arc (first order in the band width).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
from m43 import Ops, clip, frac  # noqa: E402

E = np.eye(3)
CIRCLES = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]  # (i, j, third k)


# ----------------------------------------------------------------------------------------- model-independent geometry
def _rratio(v, sigma):
    if abs(v[0]) > 1e-12:
        return v[1] / v[0]
    return np.nan if abs(v[1]) < 1e-12 else np.sign(v[1]) * sigma * np.inf


def _family_halves():
    """For each chart sigma, circle (i, j | k), side (+1: x_k max, -1: x_k min) and truth side (x_hi > x_lo): the
    spherical polygon {x_hi > x_lo} n {x_k beyond both} n {median closer to its neighbour across x_i = x_j than to
    x_k} n {sigma x0 > 0}, plus the range of r = x1/x0 over it. Fixed geometry, computed once for all models
    (not charged, like the start cones of arc_m43_repro/exact.py)."""
    out = []
    for sigma in (1.0, -1.0):
        for (i, j, k) in CIRCLES:
            for side in (1, -1):
                for hi, lo in ((i, j), (j, i)):
                    med, nb = (hi, lo) if side == 1 else (lo, hi)
                    n_true = E[hi] - E[lo]
                    n_side = (E[k] - E[hi]) if side == 1 else (E[lo] - E[k])
                    n_split = (E[k] + E[nb] - 2 * E[med]) if side == 1 else (2 * E[med] - E[nb] - E[k])
                    R = np.linalg.inv(np.array([n_true, n_side, sigma * E[0]]))
                    poly = [R[:, m] / np.linalg.norm(R[:, m]) for m in range(3)]
                    poly, _ = clip(poly, n_split)
                    rs = [r for r in (_rratio(v, sigma) for v in poly) if not np.isnan(r)]
                    out.append(dict(sigma=sigma, circle=(i, j, k), side=side, truth=med, poly=poly,
                                    n_true=n_true, rmin=min(rs), rmax=max(rs)))
    return out


HALVES = _family_halves()


# ------------------------------------------------------------------------------------------------------ helpers
def _envelope(p, q, ops):
    """Upper envelope of l_i(nu) = p_i + q_i nu on nu in [0, inf): list of (nu_lo, nu_hi, class)."""
    c = int(np.argmax(p)); lo = 0.0; out = []
    while True:
        best, nxt = np.inf, None
        for d in range(3):
            if d != c and q[d] > q[c]:
                nu = (p[c] - p[d]) / (q[d] - q[c]); ops.add(3)
                if nu > lo - 1e-12 and nu < best:
                    best, nxt = nu, d
        if nxt is None:
            out.append((lo, np.inf, c)); return out
        out.append((lo, max(best, lo), c)); lo = max(best, lo); c = nxt


def _sector(A3, pa, pb, nlo, nhi, ops):
    """Halfspaces of {nlo <= relu(A3_b x)/relu(A3_a x) <= nhi} (both-off excluded); convex."""
    cn = []
    if nlo > 0:
        cn += [A3[pb], A3[pb] - nlo * A3[pa]]; ops.add(3)
    if np.isfinite(nhi):
        cn += [A3[pa], nhi * A3[pa] - A3[pb]]; ops.add(3)
    return cn


def _pair_parts(A3, pa, pb, sect, ops):
    """(class, halfspaces) list partitioning space by the pair readout: ratio sectors + dead (-> class 0)."""
    parts = [(c, _sector(A3, pa, pb, nlo, nhi, ops)) for (nlo, nhi, c) in sect]
    parts.append((0, [-A3[pa], -A3[pb]]))
    return parts


def _err(piece, parts, truth, ops):
    """Mass of piece outside the parts of class `truth`. One correct part: sum the masses of the slivers cut off
    while clipping by its halfspaces (the clip 'outside' outputs). Otherwise: sum the wrong parts directly."""
    corr = [cn for c, cn in parts if c == truth]
    if not corr:
        return frac(piece, ops)
    if any(not cn for cn in corr):
        return 0.0
    if len(corr) == 1:
        q = piece; err = 0.0
        for n in corr[0]:
            ins, out = clip(q, n, ops)
            if out:
                err += frac(out, ops)
            q = ins
            if not q:
                break
        return err
    err = 0.0
    for c, cn in parts:
        if c == truth:
            continue
        q = piece
        for n in cn:
            q, _ = clip(q, n, ops)
            if not q:
                break
        if q:
            err += frac(q, ops)
    return err


def _strip_pieces(h, ts, ops):
    """Split the half polygon by the step-2 planes x1 = t x0 that cross it (t ascending): [(strip index, piece)]."""
    sigma = h["sigma"]
    s0 = int(np.searchsorted(ts, h["rmin"])); s1 = int(np.searchsorted(ts, h["rmax"]))
    ops.add(2 * 2)                                     # comparisons of the r-range with the thresholds
    pieces = []; cur = h["poly"]
    for s in range(s0, s1):
        ins, out = clip(cur, sigma * np.array([ts[s], -1.0, 0.0]), ops)   # inside: x1/x0 < t_s
        if ins:
            pieces.append((s, ins))
        cur = out
        if not cur:
            break
    if cur:
        pieces.append((s1, cur))
    return pieces


def _parts_for(S, A3, Wo, truth, ops):
    """Readout of a piece in which the neurons S are on (read at one point): |S| = 2 ratio sectors (relu kept for
    the pair), |S| = 1 single neuron (off -> tie -> class 0), |S| >= 3 frozen linear logits, S empty dead."""
    if len(S) == 0:
        return [(0, [])]
    if len(S) == 1:
        return [(int(np.argmax(Wo[:, S[0]])), [A3[S[0]]]), (0, [-A3[S[0]]])]
    if len(S) == 2:
        return _pair_parts(A3, S[0], S[1], _envelope(Wo[:, S[0]], Wo[:, S[1]], ops), ops)
    L = Wo[:, S] @ A3[S]; ops.add(3 * len(S) * 3 + 6)
    return [(truth, [L[truth] - L[o] for o in range(3) if o != truth])]


def _read_point(poly, n_true, tol=1e-9):
    """Midpoint of the polygon's edge on the true boundary plane (or its vertex mean if it has no such edge)."""
    on = [v for v in poly if abs(n_true @ v) < tol]
    return (np.mean(on, 0), on) if len(on) >= 2 else (np.mean(poly, 0), on)


COMPLETE = True


def _arc_err(piece, n_true, A3, Wo, truth, ops, resolve, complete=None):
    """Error in a strip piece, active set read ON the true boundary arc. resolve=True: the step-3 gate planes that
    cross the arc inside the piece are resolved (piece split by them, active set read on each sub-arc); gates that
    do not cross the arc are frozen at their value on the arc (first order in the width of the error band)."""
    complete = COMPLETE if complete is None else complete
    m, on = _read_point(piece, n_true)
    subs = [piece]
    if resolve and len(on) >= 2:
        P, Q = on[0], on[-1]
        cross = [kk for kk in range(4) if (A3[kk] @ P) * (A3[kk] @ Q) < 0]
        ops.add(7 * 4)
        for kk in cross:
            nxt = []
            for p in subs:
                a_, b_ = clip(p, A3[kk], ops)
                nxt += [q for q in (a_, b_) if q]
            subs = nxt
    err = 0.0
    for p in subs:
        mid, _ = _read_point(p, n_true) if p is not piece else (m, on)
        z = A3 @ mid; ops.add(12)
        S = [kk for kk in range(4) if z[kk] > 0]
        if complete and len(S) < 2:
            # a lone neuron is read together with the off neuron closest to switching on at this point (angular
            # distance |z_k| / |A3_k|), keeping both relus: the pair readout of the mechanism
            dist = np.abs(z) / np.linalg.norm(A3, axis=1); ops.add(4 * 4)
            off = sorted((kk for kk in range(4) if kk not in S), key=lambda kk: dist[kk])
            S = sorted(S + off[:2 - len(S)])
        err += _err(p, _parts_for(S, A3, Wo, truth, ops), truth, ops)
    return err


# ------------------------------------------------------------------------------------------------------ estimator
def est_mech(model, ops=None, info=None, strips="all", eps=0.05, resolve=("c12",), pair_rule="anchor", arc_fams=("c12",)):
    """Mechanism-derived accuracy estimate.
    strips: 'all' = every step-2 strip a family's half-chamber meets; 'anchor' = only the anchor point's strip.
    resolve: circles whose strip pieces get the arc-resolved treatment (default: the crossover x1 = x2).
    pair_rule: 'arc' = active set read on each piece's boundary arc; 'anchor' = one pair per family read at the
    anchor point (falls back to 'arc' when the anchor does not have exactly two neurons on)."""
    Wi, Wh, Wo = model
    ops = ops if ops is not None else Ops()
    err_by = {}; n_bad = 0
    for sigma in (1.0, -1.0):
        g1 = ((sigma * Wi) > 0).astype(float)
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1]); ops.add(4 * int(g1.sum()))
        t = -A2[:, 0] / Wi; ops.add(4)
        ts = np.sort(t); ops.add(5)
        bounds = [-np.inf] + list(ts) + [np.inf]
        A3s = []
        for s in range(5):
            lo, hi = bounds[s], bounds[s + 1]
            rmid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (hi - 1 if np.isfinite(hi) else lo + 1)
            g2 = (sigma * (A2[:, 0] + Wi * rmid) > 0).astype(float); ops.add(4)
            A3s.append(Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2])); ops.add(4 * int(g2.sum()) * 2)
        env_cache = {}
        if info is not None:
            info["_last"] = ops.n
        for h in HALVES:
            if h["sigma"] != sigma:
                continue
            i, j, k = h["circle"]; side = h["side"]; truth = h["truth"]
            key = (int(sigma), f"c{i}{j}{'+' if side == 1 else '-'}")
            a = np.array([sigma * eps, -side, -side]) if k == 0 else np.full(3, sigma * eps)
            if k != 0:
                a[k] = side
            sa = int(np.searchsorted(ts, a[1] / a[0])); ops.add(1 + 2)
            za = A3s[sa] @ a; ops.add(12)
            act = [m for m in range(4) if za[m] > 0]
            use_anchor = pair_rule == "anchor" and len(act) == 2 and key[1][:3] not in arc_fams
            n_bad += len(act) != 2
            if use_anchor:
                pa, pb = act
                if (pa, pb) not in env_cache:
                    env_cache[(pa, pb)] = _envelope(Wo[:, pa], Wo[:, pb], ops)
            for s, piece in _strip_pieces(h, ts, ops):
                if strips == "anchor" and s != sa:
                    continue
                A3 = A3s[s]
                res = key[1][:3] in resolve
                if use_anchor and not res:
                    e_ = _err(piece, _pair_parts(A3, pa, pb, env_cache[(pa, pb)], ops), truth, ops)
                else:
                    e_ = _arc_err(piece, h["n_true"], A3, Wo, truth, ops, res)
                err_by[key] = err_by.get(key, 0.0) + e_
            if info is not None:
                d_ = info.setdefault("ops_by", {}); d_[key] = d_.get(key, 0) + ops.n - info.get("_last", 0); info["_last"] = ops.n
    if info is not None:
        info["err_by"] = err_by; info["bad_pairs"] = n_bad
    return 1.0 - sum(err_by.values())

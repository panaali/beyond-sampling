"""T63: accuracy of M_4,3 from its mechanism ("pair-ratio boundary account").

Mechanism (algzoo/labels_4_3.py, derived here from the weights, see README "What the network does"):
  * step 1: sign(x0) picks which mirror half carries x0 (chart sigma = sign x0).
  * step 2: the four step-2 gates switch on the planes x1 = t_k x0 (t_k = -A2_k0 / Wi_k, closed form); they cut
    each chart into 5 "strips" in r = x1/x0. The middle strip, which contains x1 = x0, is the soft x1-vs-x0 switch.
  * step 3: every neuron is h_k = ReLU(A3_k . x), A3 = Wh diag(g2) A2 + Wi e2^T (closed form per strip).
  * readout: near each true decision boundary only TWO neurons (a, b) are active (the u pair {0,1} when x2 is
    above the two compared inputs, the v pair {2,3} when below, a crossover pair {outer, inner-mirror} on x1 = x2),
    so the logits are l_i = Wo[i,a] h_a + Wo[i,b] h_b and the class is a function of the RATIO nu = h_b / h_a:
    the upper envelope of three lines in nu gives thresholds nu_1 < nu_2 and the class order (closed form).
    Both-off ("dead") gives the argmax tie -> class 0.
Error families: for each true boundary circle x_i = x_j (i<j), side (third coordinate k max or min), chart sigma
and strip, the error is the union of convex cones {chamber half} n {strip} n {ratio sector of a wrong class}.
Each chamber is split between its two adjacent boundaries by the "median closer to which neighbour" plane, so the
families are disjoint. Accuracy = 1 - sum of cone masses (solid angles, Van Oosterom-Strackee = Girard).

No model outputs are fitted and no enumeration of step-3 gate patterns is done: the pair for each family is read
off at one anchor point near the far (x0 -> 0) end of the boundary arc.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
from m43 import Ops, clip, frac  # noqa: E402

E = np.eye(3)
CIRCLES = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]  # (i, j, third k)


def _gates2(Wi, A2, sigma, r):
    """Step-2 gate pattern at ratio r = x1/x0 in chart sigma (A2_k . x has the sign of sigma*(A2_k0 + Wi_k r))."""
    return (sigma * (A2[:, 0] + Wi * r) > 0).astype(float)


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


def _cone(normals, ops):
    """Gaussian mass of {x: n.x >= 0 for all n}; first three normals must be independent (used for checks)."""
    R = np.linalg.inv(np.array(normals[:3]))
    poly = [R[:, i] / np.linalg.norm(R[:, i]) for i in range(3)]
    for n in normals[3:]:
        poly, _ = clip(poly, np.asarray(n, float), ops)
        if not poly:
            return 0.0
    return frac(poly, ops)


def _family_halves():
    """Model-independent geometry: for each chart sigma, boundary circle (i, j | k), side (+: x_k max, -: x_k min)
    and truth side (x_hi > x_lo), the spherical polygon of the half-chamber assigned to this boundary:
    {x_hi > x_lo} n {x_k beyond both} n {median closer to its neighbour across x_i = x_j than to x_k} n {sigma x0 > 0}.
    Computed once (no model dependence), like the 12 simplicial start cones of the exact enumerator."""
    out = []
    for sigma in (1.0, -1.0):
        for (i, j, k) in CIRCLES:
            for side in (1, -1):
                for hi, lo in ((i, j), (j, i)):
                    med, nb = (hi, lo) if side == 1 else (lo, hi)
                    n_true = E[hi] - E[lo]
                    n_side = (E[k] - E[hi]) if side == 1 else (E[lo] - E[k])
                    n_split = (E[k] + E[nb] - 2 * E[med]) if side == 1 else (2 * E[med] - E[nb] - E[k])
                    R = np.linalg.inv(np.array([n_true, n_side, n_split + 1e-9 * E[0]])) if False else None
                    # pointed start: true plane, side plane, chart plane; then clip by the split plane
                    R = np.linalg.inv(np.array([n_true, n_side, sigma * E[0]]))
                    poly = [R[:, m] / np.linalg.norm(R[:, m]) for m in range(3)]
                    poly, _ = clip(poly, n_split)
                    truth = hi if side == 1 else lo
                    out.append((sigma, (i, j, k), side, truth, poly))
    return out


HALVES = _family_halves()


def _strip_pieces(poly, sigma, ts, ops):
    """Split a polygon (inside chart sigma) by the step-2 planes x1 = t x0 (t ascending): returns list of
    (strip index, piece). Strip s is t_(s) < x1/x0 < t_(s+1)."""
    pieces = []
    cur = poly
    for s, tv in enumerate(ts):
        if not cur:
            break
        n = sigma * np.array([tv, -1.0, 0.0])         # inside: x1/x0 < t
        ins, out = clip(cur, n, ops)
        if ins:
            pieces.append((s, ins))
        cur = out
    if cur:
        pieces.append((len(ts), cur))
    return pieces


def _sector(A3, pa, pb, nlo, nhi):
    """Halfspaces of {nlo <= relu(A3_b x)/relu(A3_a x) <= nhi} (both-off excluded)."""
    cn = []
    if nlo > 0:
        cn += [A3[pb], A3[pb] - nlo * A3[pa]]
    if np.isfinite(nhi):
        cn += [A3[pa], nhi * A3[pa] - A3[pb]]
    return cn


def _mass_in(poly, normals, ops):
    """(mass of poly n halfspaces, unchanged flag)."""
    q = poly; same = True
    for n in normals:
        ins, out = clip(q, n, ops)
        if out:
            same = False
        q = ins
        if not q:
            return 0.0, False
    return (None, True) if same else (frac(q, ops), False)


def est_mech(model, ops=None, info=None, strips="all", eps=0.05):
    """Mechanism-derived accuracy estimate (see module docstring).
    strips='all': every step-2 strip met by a family's half-chamber; 'anchor': only the anchor's strip."""
    Wi, Wh, Wo = model
    ops = ops if ops is not None else Ops()
    err_by = {}; n_bad = 0
    for sigma in (1.0, -1.0):
        g1 = ((sigma * Wi) > 0).astype(float)
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1]); ops.add(4 * int(g1.sum()))
        t = -A2[:, 0] / Wi; ops.add(4)
        ts = np.sort(t)
        bounds = [-np.inf] + list(ts) + [np.inf]
        A3s = []
        for s in range(5):
            lo, hi = bounds[s], bounds[s + 1]
            rmid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (hi - 1 if np.isfinite(hi) else lo + 1)
            g2 = _gates2(Wi, A2, sigma, rmid); ops.add(4)
            A3s.append(Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2])); ops.add(4 * int(g2.sum()) * 2)
        env_cache = {}
        for (sg, (i, j, k), side, truth, poly) in HALVES:
            if sg != sigma:
                continue
            a = np.array([sigma * eps, -side, -side]) if k == 0 else np.full(3, sigma * eps)
            if k != 0:
                a[k] = side
            ra = a[1] / a[0]
            sa = int(np.searchsorted(ts, ra))
            za = A3s[sa] @ a; ops.add(12)
            act = [m for m in range(4) if za[m] > 0]
            if len(act) != 2:
                n_bad += 1
                act = list(np.argsort(-za)[:2])
            pa, pb = sorted(int(m) for m in act)
            if (pa, pb) not in env_cache:
                env_cache[(pa, pb)] = _envelope(Wo[:, pa], Wo[:, pb], ops)
            sect = env_cache[(pa, pb)]
            key = (int(sigma), f"c{i}{j}{'+' if side == 1 else '-'}")
            for s, piece in _strip_pieces(poly, sigma, ts, ops):
                if strips == "anchor" and s != sa:
                    continue
                A3 = A3s[s]
                correct = [_sector(A3, pa, pb, nlo, nhi) for (nlo, nhi, c) in sect if c == truth]
                if truth == 0:
                    correct.append([-A3[pa], -A3[pb]])          # both off: argmax of zeros -> class 0
                ops.add(6 * sum(len(c) for c in correct))
                got = 0.0; whole = False
                for cn in correct:
                    m, same = _mass_in(piece, cn, ops)
                    if same:
                        whole = True; break
                    got += m
                if whole:
                    continue
                err = frac(piece, ops) - got
                err_by[key] = err_by.get(key, 0.0) + err
    if info is not None:
        info["err_by"] = err_by; info["bad_pairs"] = n_bad
    return 1.0 - sum(err_by.values())

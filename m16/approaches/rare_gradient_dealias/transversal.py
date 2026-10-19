"""T386 family T: the same closure estimand as K3 (T340c / T354 lines, lows pinned as functions of v3), integrated with
lines along g12 (the top input v1 = v2 + g12) instead of along g23. Given v2, g12 and g23 are independent
(v1 | v1 > v2, v3 = max of the 8 inputs below v2), so
  E1 = sum_v2 w(v2) 90 phi(v2) sum_{t nodes} tw(v2) (1/NT) sum_triples int_{g12 : p3 is argmax} phi(v2 + g12) dg12,
with the g23 rule (nodes t, weights tw) a K_t-node Gauss rule for the density 8 Phi(v2 - t)^7 phi(v2 - t) on
[t0, T_hi] (weights sum to Phi(v2 - t0)^8 - Phi(v2 - T_hi)^8) and the g12-integral exact along each line
(g12 in [0, G_HI]). The gradient is T372's boundary term with line density phi(v2 + g12) and dm/dg12 = dm/dx_p1.
Why: at the K3 endpoints the margin changes ~9x faster per unit g12 than per unit g23 (scale.py), so the islands
are thin in g12 and the g23-lines graze them (near-tangent chords, aliasing); g12-lines cross them transversally.
v2 rule as dealias.v2_rule (gl20 = K3's). Deductive: fixed model-free lines, no random numbers.
est_grad(model, t0, Kt, v2spec) -> dict(est, grad, C_walk, C_grad, n_end)."""
import numpy as np
from scipy.special import ndtr

import dealias
import t372_lgrad as lgrad
import t372_walk as walk

TMAX = 6.0; G_HI = 8.0
G, NT, TRI, K = dealias.G, dealias.NT, dealias.TRI, dealias.K


def t_rule(v2, t0, Kt, n=40001):
    """Kt-node Gauss rule on [t0, TMAX] for the density 8 Phi(v2 - t)^7 phi(v2 - t) (Stieltjes on a fine model-free grid);
    weights sum to the exact mass Phi(v2 - t0)^8 - Phi(v2 - TMAX)^8."""
    t = np.linspace(t0, TMAX, n); w = 8 * ndtr(v2 - t) ** 7 * G._pdf(v2 - t); w = w / w.sum()
    a, b = np.zeros(Kt), np.zeros(Kt); p_prev = np.zeros_like(t); p = np.ones_like(t); nrm_prev = 1.0
    for j in range(Kt):
        nrm = (w * p * p).sum(); a[j] = (w * t * p * p).sum() / nrm
        if j > 0:
            b[j] = nrm / nrm_prev
        p_new = (t - a[j]) * p - (b[j] if j > 0 else 0) * p_prev
        p_prev, p, nrm_prev = p, p_new, nrm
    J = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)
    x, U = np.linalg.eigh(J)
    return x, U[0] ** 2 * (ndtr(v2 - t0) ** 8 - ndtr(v2 - TMAX) ** 8)


def lines(v2, t0, Kt):
    """Rows (t node, triple): base point at g12 = 0 on the K3 closure line through (v2, g23 = t), direction e_p1."""
    v3s = v2 - K.rare._cond_g23(v2); v4s = float(G.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
    m4 = float(G.mu(np.array([v4s]))[0]); dm4 = float(G.dmu(v4s)[0])
    x23, dx23 = G.inputs(v2, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
    tn, tw = t_rule(v2, t0, Kt); e1 = np.zeros((NT, 10)); e1[np.arange(NT), TRI[:, 0]] = 1.0
    x = np.concatenate([x23 + t * dx23 for t in tn]); dx = np.tile(e1, (Kt, 1)); om = np.repeat(tw, NT)
    return x, dx, om


def est_grad(model, t0, Kt=3, v2spec="gl20", chunk_rows=45000, want_grad=True):
    model = tuple(np.asarray(w, float) for w in model)
    est = 0.0; grad = np.zeros(sum(w.size for w in model)); Cw = 0.0; Cg = 0.0; n_end = 0
    for name, lo, hi in G.REGIONS:
        vn, vw = dealias.v2_rule(v2spec, lo, hi); wgt = 90 * G._pdf(vn) * vw
        per = Kt * NT; step = max(1, chunk_rows // per)
        for s in range(0, len(vn), step):
            js = np.arange(s, min(s + step, len(vn))); L = [lines(vn[j], t0, Kt) for j in js]
            x = np.concatenate([l[0] for l in L]); dx = np.concatenate([l[1] for l in L]); om = np.concatenate([l[2] for l in L])
            nrow = len(x); nr = np.repeat(np.arange(len(js)), per); v2 = vn[js][nr]
            k3 = np.tile(TRI[:, 2], nrow // NT)
            pieces, _, f = K.T354.walk_seg(model, x, dx, np.tile(TRI[:, 1], nrow // NT), k3, 0.0, G_HI); Cw += f
            R = np.concatenate([p[0] for p in pieces]); A = np.concatenate([p[1] for p in pieces]); B = np.concatenate([p[2] for p in pieces])
            m = B > A; R, A, B = walk.merge(R[m], A[m], B[m]); Cw += len(R) * 2 * (G.F_PHI + 3)
            rowfac = om * wgt[js][nr] / NT
            est += float((rowfac[R] * (ndtr(v2[R] + B) - ndtr(v2[R] + A))).sum())
            if not want_grad:
                continue
            for e, keep in ((A, A > 1e-9), (B, B < G_HI - 1e-9)):
                r, ee = R[keep], e[keep]
                if not len(r):
                    continue
                Gm, J, S, f = lgrad.endpoint_grads(model, x, dx, k3, r, ee); Cg += f
                grad += ((rowfac[r] * G._pdf(v2[r] + ee))[:, None] * Gm).sum(0); n_end += len(r)
    return dict(est=est, grad=grad, C_walk=Cw / K.FPE, C_grad=Cg / K.FPE, n_end=n_end)

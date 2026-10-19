"""T386 family S: smoothed-indicator line rule. Along each K3-type line x(t) = x0 + t dx (t = g23 in [t0, T_hi]), the
hard indicator {rank-3 position is the argmax} = {m*(t) > 0}, m* = L_p3 - max_{j != p3} L_j, is replaced by
S(m*/eps) with S the integrated biweight (S' = 15/16 (1 - u^2)^2 on |u| < 1). The line mass
int f(t) S(m*(t)/eps) dt (f = d/dt [-Phi(v2 - t)^8], as rare.mass) and its exact W-derivative
int f S'(m*/eps)/eps dm*/dW dt are computed per gate cell: m* is concave piecewise affine there, the band
{|m*| < eps} is found exactly, and the band integral uses Q Gauss-Legendre points per band piece (split at the
competitor switches). The region {m* >= eps} contributes its exact mass and no gradient. eps -> 0 gives the hard
K3 estimate and T372's boundary gradient. Deductive: fixed model-free lines, no random numbers.
est_grad(model, t0, eps, v2spec, g12spec, Q) -> dict(est, grad, C_walk, C_grad, n_pts, scale)."""
import numpy as np
from scipy.special import ndtr

import dealias
import t372_rnn as rnn
import t372_walk as walk

TMAX = 6.0
G, NT, TRI = dealias.G, dealias.NT, dealias.TRI


def S(u):
    u = np.clip(u, -1, 1); return 0.5 + (15 * u - 10 * u ** 3 + 3 * u ** 5) / 16


def dS(u):
    return np.where(np.abs(u) < 1, 15 / 16 * (1 - u ** 2) ** 2, 0.0)


def _cells(model, x0, d, lo, hi, max_cells=20000):
    """Exact walk (T372 walk.py's loop): per gate cell, (rows, v_start, v_end, L(v_start), dL)."""
    n = len(x0); v = lo.astype(float).copy(); act = np.arange(n); out = []; flops = 0.0; it = 0
    Wi, Wh, Wo = model; T = x0.shape[1]; H = len(Wi); C = len(Wo)
    fl_cell = T * (4 * H * H + 6 * H + 6 * H) + 4 * C * H + 6 * C + T * H * 4
    while len(act) and it < max_cells:
        it += 1
        X = x0[act] + v[act, None] * d[act]; Z, DZ, L, dL = walk._fwd_tan(model, X, d[act])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -Z / DZ
        r = np.where(np.isfinite(r) & (r > 1e-12 * (1 + np.abs(v[act]))[:, None, None]), r, np.inf).reshape(len(act), -1).min(1)
        nxt = np.minimum(v[act] + r, hi[act]); out.append((act, v[act].copy(), nxt, L, dL)); flops += len(act) * fl_cell
        v[act] = nxt; act = act[nxt < hi[act]]
    assert not len(act)
    return out, flops


def _env_pieces(c, e, k, length):
    """Within one cell (offset s in [0, length]), m*(s) = min_{j != k} (c_j + e_j s) with c, e the margins L_k - L_j.
    Split by the active competitor: returns [(s_a, s_b, j)]."""
    js = [jj for jj in range(len(c)) if jj != k]
    j = min(js, key=lambda jj: (c[jj], e[jj])); s = 0.0; out = []
    while s < length:
        best, nj = length, None
        for jj in js:
            if jj != j and e[jj] < e[j]:
                x = (c[jj] - c[j]) / (e[j] - e[jj])
                if s < x < best:
                    best, nj = x, jj
        out.append((s, best, j)); s = best
        if nj is None:
            break
        j = nj
    return out


def est_grad(model, t0, eps, v2spec="gl20", g12spec=("gauss", 3), Q=6, chunk_rows=45000):
    model = tuple(np.asarray(w, float) for w in model); P = sum(w.size for w in model)
    gq, wq = np.polynomial.legendre.leggauss(Q)
    est = 0.0; grad = np.zeros(P); Cw = 0.0; Cg = 0.0; npts = 0; T, H, C = 10, len(model[0]), len(model[2])
    for name, lo, hi in G.REGIONS:
        vn, vw = dealias.v2_rule(v2spec, lo, hi)
        wgt = 90 * G._pdf(vn) * (1.0 - ndtr(vn)) * vw
        per = len(dealias.g12_rule(g12spec, vn[0])[0]) * NT; step = max(1, chunk_rows // per)
        for s0 in range(0, len(vn), step):
            js = np.arange(s0, min(s0 + step, len(vn))); Ls = [dealias.lines(vn[j], g12spec) for j in js]
            x = np.concatenate([l[0] for l in Ls]); dx = np.concatenate([l[1] for l in Ls]); om = np.concatenate([l[2] for l in Ls])
            nrow = len(x); v2 = vn[js][np.repeat(np.arange(len(js)), per)]; k3 = np.tile(TRI[:, 2], nrow // NT)
            rowfac = om * wgt[js][np.repeat(np.arange(len(js)), per)] / NT
            cells, f = _cells(model, x, dx, np.full(nrow, t0), np.full(nrow, TMAX)); Cw += f
            pr, pa, pb, pj = [], [], [], []
            for rows, va, vb, L, dL in cells:
                k = k3[rows]; rr = np.arange(len(rows)); length = vb - va
                c = L[rr, k][:, None] - L; e = dL[rr, k][:, None] - dL; c[rr, k] = np.inf; e[rr, k] = 0.0
                # m* is concave on the cell and <= min_j max(c_j, c_j + e_j len): screen cells that cannot reach -eps
                ub = np.maximum(c, c + e * length[:, None]).min(1)
                for i in np.nonzero(ub > -eps)[0]:
                    for sa, sb, j in _env_pieces(c[i] + 0.0, e[i] + 0.0, k[i], length[i]):
                        # on [sa, sb] m*(s) = c_j + e_j s (affine): split at m* = +-eps
                        cj, ej = c[i, j], e[i, j]
                        cuts = [sa, sb] + [(-cj + q) / ej for q in (-eps, eps) if ej != 0 and sa < (-cj + q) / ej < sb]
                        cuts = sorted(cuts)
                        for a, b in zip(cuts[:-1], cuts[1:]):
                            mm = cj + ej * (a + b) / 2
                            if mm <= -eps or b <= a:
                                continue
                            ta, tb = va[i] + a, va[i] + b
                            if mm >= eps:                                   # interior: exact mass, no gradient
                                est += rowfac[rows[i]] * (ndtr(v2[rows[i]] - ta) ** 8 - ndtr(v2[rows[i]] - tb) ** 8)
                            else:
                                pr.append(rows[i]); pa.append(ta); pb.append(tb); pj.append(j)
            if not pr:
                continue
            pr, pa, pb, pj = map(np.array, (pr, pa, pb, pj))
            tt = (pa[:, None] + (pb - pa)[:, None] * (gq[None] + 1) / 2).ravel(); ww = ((pb - pa)[:, None] / 2 * wq[None]).ravel()
            rr = np.repeat(pr, Q); jj = np.repeat(pj, Q); X = x[rr] + tt[:, None] * dx[rr]
            fdens = 8 * ndtr(v2[rr] - tt) ** 7 * np.exp(-(v2[rr] - tt) ** 2 / 2) / np.sqrt(2 * np.pi)
            for c0 in range(0, len(rr), 20000):
                sl = slice(c0, c0 + 20000)
                m, gw, _ = rnn.margin_grad(model, X[sl], k3[rr[sl]], jj[sl])
                u = m / eps; base = rowfac[rr[sl]] * fdens[sl] * ww[sl]
                est += float((base * S(u)).sum()); grad += ((base * dS(u) / eps)[:, None] * gw).sum(0)
            npts += len(rr); Cg += len(rr) * (rnn.flops_per_grad(T, H, C) + 2 * T + 2 * C * H)
    return dict(est=float(est), grad=grad, C_walk=Cw / dealias.K.FPE, C_grad=Cg / dealias.K.FPE, n_pts=npts)

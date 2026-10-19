"""T449 candidate (a): weight-derived switch-point sensitivity along each frozen line.

On a linear piece of the line x(t) = w + t d the network is x -> J x (bias-free ReLU RNN, gate pattern fixed). A
switch point r solves c . (w + r d) = 0, with c = a pre-activation row (gate switch) or J_j - J_k (an endpoint of the
wrong stretch, k = the correct output). Moving the line by v perpendicular to d moves r by -(c . v) / (c . d), so
    rho = |c_perp| / |c . d|          (|dr/dw|, exact from the weights)
and |grad_w g| <= sum_endpoints k(r) rho + (window terms). rho diverges where the line grazes the boundary (c . d -> 0).

usage: python sens.py MODEL [MODEL ...]  -> out/sens_<model>.npz (per stratum: endpoint rho, k(r), gate rho) + check
"""
import os
import sys

import numpy as np

import lib
import hom

N = int(os.environ.get("SENS_N", 1024))


def jac_pattern(model, x):
    """Gate pattern at x (T,) -> pre-activation maps A (T, H, T) and output map J (O, T)."""
    Wi, Wh, Wo = model
    T = len(x); H = Wh.shape[0]
    G = np.zeros((H, T)); A = np.zeros((T, H, T))
    h = np.zeros(H)
    for s in range(T):
        z = Wh @ h + Wi * x[s]
        Z = Wh @ G; Z[:, s] += Wi
        A[s] = Z
        on = z > 0
        h = np.where(on, z, 0.0); G = Z * on[:, None]
    return A, Wo @ G


def line(model, w, a, b, tau, kind, cdf):
    T = len(w)
    d = np.zeros(T); d[a] = 1 / np.sqrt(2); d[b] = -1 / np.sqrt(2)
    tpos, tneg = (a, b) if kind == "23" else (b, a)
    P = np.eye(T) - np.outer(d, d)
    t = -tau; mass = 0.0; ends = []; gates = []
    while t < tau:
        tc = t + 1e-10 * (1 + abs(t))
        A, J = jac_pattern(model, w + tc * d)
        z0 = A @ w; dz = A @ d                                   # (T, H)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -z0 / dz
        ok = (r > tc) & np.isfinite(r)
        nxt = min(r[ok].min() if ok.any() else np.inf, tau)
        if ok.any():
            s, i = np.unravel_index(np.where(ok, r, np.inf).argmin(), r.shape)
            if r[s, i] < tau:
                c = A[s, i]; gates.append(np.linalg.norm(P @ c) / max(abs(c @ d), 1e-300))
        for lo, hi, k in ((t, min(nxt, 0.0), tneg), (max(t, 0.0), nxt, tpos)):
            if hi <= lo:
                continue
            L0 = J @ w; dL = J @ d
            # correct set on [lo, hi]: L_k - L_j > 0 for all j != k (ties ignored, measure zero)
            cl, ch = lo, hi; who = [None, None]
            for j in range(len(L0)):
                if j == k:
                    continue
                cc, ee = L0[k] - L0[j], dL[k] - dL[j]
                if ee == 0:
                    if cc <= 0:
                        cl, ch = 1, 0
                    continue
                root = -cc / ee
                if ee > 0 and root > cl:
                    cl, who[0] = root, j
                if ee < 0 and root < ch:
                    ch, who[1] = root, j
            corr = cdf(ch) - cdf(cl) if ch > cl else 0.0
            mass += (cdf(hi) - cdf(lo)) - corr
            if ch > cl:
                for side, rt in ((0, cl), (1, ch)):
                    j = who[side]
                    if j is not None and lo < rt < hi:
                        c = J[k] - J[j]
                        ends.append((np.linalg.norm(P @ c) / max(abs(c @ d), 1e-300), rt))
        t = nxt
    return mass, ends, gates


def run(name, model):
    out = {}
    for j, (kind, h) in enumerate(lib.STRATA):
        U = lib.node(j, N)
        y, a, b, tau, wt = lib.E.face_points_strat(lib.T, kind, h, U)
        r = np.linalg.norm(y, axis=1); w = y / r[:, None]; tw = np.minimum(tau / r, 1e8)
        cdf = hom.tcdf(lib.T)
        ref = lib.g(model, j, U)[0]
        nu = lib.T - 1; sq = np.sqrt(nu)
        from scipy.stats import t as tdist
        dens = lambda x: tdist.pdf(x * sq, nu) * sq
        masses, er, ek, gr, gl = [], [], [], [], []
        for i in range(N):
            m, ends, gates = line(model, w[i], a[i], b[i], tw[i], kind, cdf)
            masses.append(m * wt[i])
            for rho, rt in ends:
                er.append(rho); ek.append(dens(rt)); gl.append(i)
            gr.extend(gates)
        masses = np.array(masses)
        out[f"rho_end_{j}"] = np.array(er); out[f"k_end_{j}"] = np.array(ek); out[f"line_{j}"] = np.array(gl)
        out[f"rho_gate_{j}"] = np.array(gr); out[f"check_{j}"] = np.abs(masses - ref).max()
        print(name, j, "max |mass - frozen| %.2e" % out[f"check_{j}"], "endpoints", len(er), flush=True)
    np.savez(os.path.join(lib.HERE, "out", f"sens_{name}.npz"), **out)


if __name__ == "__main__":
    ms, _ = lib.models()
    for n in sys.argv[1:]:
        run(n, ms[n])

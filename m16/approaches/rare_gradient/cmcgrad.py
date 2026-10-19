"""T372 secondary (SAMPLING-CLASS, unbiased): conditional MC along exact v3 lines for E1(t0) and its weight-gradient.
x is drawn from its exact law given g23 > t0 (T364's sampler; v3 then has density phi on (v4, v2 - t0) given the rest),
v3 is integrated out exactly by walking x_p3 = v over (v4, v2 - t0) with walk.py (independent of rare.walk3):
Z = mass of {argmax = p3} / (Phi(v2 - t0) - Phi(v4)), P = P(g23 > t0) E[Z], dP/dW = P(g23 > t0) E[dZ/dW] with dZ/dW
from the boundary terms (lgrad). Reports P, the gradient (432) and its projections on the 6 frozen directions, with
s.e. usage: python cmcgrad.py model n_lines -> results/cmc_<model>.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

import lgrad
import rnn
import truthc_copy as TCC
import walk

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = {"orig": 372200, "dev3_s526": 372201, "dev3_s528": 372202}


def draw(c, t0, rng, n):
    v2 = np.interp(rng.random(n), c, TCC.GRID); v1 = -ndtri(rng.random(n) * ndtr(-v2))
    v3 = ndtri(ndtr(v2 - t0) * rng.random(n) ** (1 / 8)); lows = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))
    vals = np.column_stack([v1, v2, v3, lows]); perm = np.argsort(rng.random((n, 10)), 1)
    x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
    return x, perm, v2, lows.max(1)


def main(nm, nlines, chunk=20_000):
    D = json.load(open(os.path.join(HERE, "results", f"dirs_{nm}.json"))); t0 = D["t0"]
    model = rnn.unflat(np.array(D["w"]), [(16,), (16, 16), (10, 16)]); U = np.array(D["dirs"]); P_ = U.shape[1]
    f = TCC.dens(t0); c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(TCC.GRID))]); c /= c[-1]
    pg = TCC.p_gap(t0); rng = np.random.default_rng(SEED[nm])
    sZ = sZ2 = 0.0; sg = np.zeros(P_); sg2 = np.zeros(P_); sp = np.zeros(len(U)); sp2 = np.zeros(len(U))
    flops = 0.0; n = 0; nz = 0; n_end = 0; t = time.time()
    while n < nlines:
        m = min(chunk, nlines - n); x, perm, v2, v4 = draw(c, t0, rng, m); p3 = perm[:, 2]; r = np.arange(m)
        d = np.zeros_like(x); d[r, p3] = 1.0; x0 = x.copy(); x0[r, p3] = 0.0
        lo, hi = v4, v2 - t0; nrm = ndtr(hi) - ndtr(lo)
        R, A, B, fl = walk.argmax_pieces(model, x0, d, p3, lo, hi); flops += fl
        R, A, B = walk.merge(R, A, B)
        Z = np.zeros(m); np.add.at(Z, R, (ndtr(B) - ndtr(A)) / nrm[R]); flops += len(R) * 50
        Gl = np.zeros((m, P_))
        for e, keep in ((A, A > lo[R] + 1e-9 * (1 + np.abs(lo[R]))), (B, B < hi[R] - 1e-9 * (1 + np.abs(hi[R])))):
            rr, ee = R[keep], e[keep]
            if len(rr):
                Gm, _, _, fl = lgrad.endpoint_grads(model, x0, d, p3, rr, ee); flops += fl; n_end += len(rr)
                np.add.at(Gl, rr, (np.exp(-ee ** 2 / 2) / np.sqrt(2 * np.pi) / nrm[rr])[:, None] * Gm)
        sZ += Z.sum(); sZ2 += (Z ** 2).sum(); sg += Gl.sum(0); sg2 += (Gl ** 2).sum(0)
        pr = Gl @ U.T; sp += pr.sum(0); sp2 += (pr ** 2).sum(0); nz += int((Z > 0).sum()); n += m
        se = lambda s, s2: np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 0) / n)
        out = dict(model=nm, t0=t0, n_lines=n, p_gap=pg, P=pg * sZ / n, P_se=pg * se(sZ, sZ2), frac_nonzero=nz / n,
                   grad=(pg * sg / n).tolist(), grad_se=(pg * se(sg, sg2)).tolist(), proj=(pg * sp / n).tolist(), proj_se=(pg * se(sp, sp2)).tolist(),
                   n_end=n_end, C_fpe=flops / 5948.0, sec=time.time() - t)
        json.dump(out, open(os.path.join(HERE, "results", f"cmc_{nm}.json"), "w"))
        print(nm, n, "P %.4e +- %.1e" % (out["P"], out["P_se"]), "proj/P", np.round(np.array(out["proj"]) / out["P"], 3),
              "z", np.round(np.array(out["proj"]) / np.maximum(out["proj_se"], 1e-300), 1), "C %.0f" % out["C_fpe"], "%.0fs" % out["sec"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(float(sys.argv[2])))

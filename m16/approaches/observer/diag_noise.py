"""DIAGNOSTIC (uses random rest inputs): is the 'internal noise' model right?
Fix top-3 at an anchor (tie kind 23, level v, gap12 g, relative 2-3 gap r); draw the 7 rest inputs from their LHS
strata (q=2 bins of N(0,1)|<c); compare the true P(network picks rank-3) with the observer predictions:
  lin   : Gaussian, exact local map, within-bin variance (v3, noise=1)
  clip  : Gaussian, map applied to clip(x_t, cone interval) (v4)
  point : no noise, anchor's own correctness (v3 noise=0)"""
import sys, numpy as np
from scipy.special import ndtr
from scipy.stats import norm
sys.path.insert(0, '/home/user/clean-neurons/m16')
from core import T, TRIPLES, map_and_cone, bin_moments, clipped_bin_moments, forward_gates
from v1 import tie_levels
from v3 import REST
from common import Wi, Wh, Wo
model = (Wi, Wh, Wo); q = 2; S = 200
vq, ev, gq, eg = tie_levels('23', 1, 1); v, og = vq[0], gq[0, 0]
rng = np.random.default_rng(5)
Pc = None
for r in (0.005, 0.018, 0.05, 0.13):
    a, b, c = v + og, v, v * (1 - r); Pc = ndtr(c)
    med = norm.ppf(Pc * (np.arange(q) + 0.5) / q)
    lhs = rng.integers(0, q, (720, 7))
    X = np.empty((720, T)); X[np.arange(720), TRIPLES[:, 0]] = a; X[np.arange(720), TRIPLES[:, 1]] = b
    X[np.arange(720), TRIPLES[:, 2]] = c; X[np.arange(720)[:, None], REST] = med[lhs]
    _, Am, LO, HI = map_and_cone(model, X)
    ii = np.arange(720); P1, P2, P3 = TRIPLES.T
    D = Am[ii, P2] - Am[ii, P3]; dr = np.take_along_axis(D, REST, 1)
    BM, BV = bin_moments(np.array([c]), q)
    mu_lin = D[ii, P1] * a + D[ii, P2] * b + D[ii, P3] * c + (dr * BM[lhs, 0]).sum(1)
    sd_lin = np.sqrt((dr ** 2 * BV[lhs, 0]).sum(1))
    lo_r = np.take_along_axis(LO, REST, 1); hi_r = np.take_along_axis(HI, REST, 1)
    EM = np.zeros((720, 7)); EV = np.zeros((720, 7))
    for rb in range(q):
        sel = lhs == rb; EM[sel], EV[sel] = clipped_bin_moments(c, q, rb, lo_r[sel], hi_r[sel])
    mu_c = D[ii, P1] * a + D[ii, P2] * b + D[ii, P3] * c + (dr * EM).sum(1); sd_c = np.sqrt((dr ** 2 * EV).sum(1))
    point = (D * X).sum(1) < 0
    # truth by sampling rest within strata
    U = (lhs[:, None, :] + rng.random((720, S, 7))) / q
    Xs = np.repeat(X[:, None], S, 1); Xs[np.arange(720)[:, None, None], np.arange(S)[None, :, None], REST[:, None, :]] = norm.ppf(Pc * U)
    L, _ = forward_gates(model, Xs.reshape(-1, T)); out = L.argmax(1).reshape(720, S)
    true = (out == P3[:, None]).mean(1)
    pl = ndtr(-mu_lin / sd_lin); pc = ndtr(-mu_c / sd_c)
    print(f'r={r}: true {true.mean():.3f} | lin {pl.mean():.3f} clip {pc.mean():.3f} point {point.mean():.3f} | '
          f'corr lin {np.corrcoef(true, pl)[0,1]:.2f} clip {np.corrcoef(true, pc)[0,1]:.2f} point {np.corrcoef(true, point)[0,1]:.2f}'
          f' | MAE lin {np.abs(true-pl).mean():.3f} clip {np.abs(true-pc).mean():.3f} point {np.abs(true-point).mean():.3f}'
          f' | frac true in (0.05,0.95) {((true>0.05)&(true<0.95)).mean():.2f}', flush=True)

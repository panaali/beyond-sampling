"""Diagnosis of 'linear response of the band edge': at each face point y0, take the exact signed band delta(y0)
and its slope beta wrt the 8 free face coordinates (max o1 + 7 bath), then average the adjacent band mass
Phi0(|delta0 + beta.(y'-y0)|) over the face law y' | (config, v). Compare the node-average of that (B) to the
node-average of the exact mass (A). Random points used for diagnosis only.
usage: python diag_linavg.py <model> [N] [h]"""
import sys
import numpy as np
from scipy.special import ndtr, ndtri
import wtool as W

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
M = W.get_model(nm)
rng = np.random.default_rng(21)
y, a, b, p1, v, o1, o2 = W.surface(N, rng)
rows = np.arange(N)
free = np.ones((N, 10), bool); free[rows, a] = False; free[rows, b] = False
fidx = np.nonzero(free)[1].reshape(N, 8)


def run(yy):
    oth = yy.copy(); oth[rows, a] = -np.inf; oth[rows, b] = -np.inf
    s_ = -np.sort(-oth, 1); vv = yy[rows, a]
    tau = W.SQ2 * np.clip(np.minimum(s_[:, 0] - vv, vv - s_[:, 1]), 1e-9, None)
    return W.walk(M, yy, a, b, tau)


R0 = run(y); d0 = R0["delta"]; mass = R0["mass"]; madj = R0["madj"]
beta = np.zeros((N, 8))
for j in range(8):
    yp = y.copy(); ym = y.copy()
    col = fidx[:, j]
    # one-sided where the step would cross the tie value or swap the max below v
    isMax = col == p1
    up = y[rows, col] + h; dn = y[rows, col] - h
    up = np.where(~isMax & (up >= v - 1e-6), y[rows, col], up)
    dn = np.where(isMax & (dn <= v + 1e-6), y[rows, col], dn)
    yp[rows, col] = up; ym[rows, col] = dn
    beta[:, j] = (run(yp)["delta"] - run(ym)["delta"]) / np.maximum(up - dn, 1e-12)
# average over y' | (config, v) by MC (diagnosis): K draws per node
K = 64
ONLY = sys.argv[4] if len(sys.argv) > 4 else "all"
if ONLY == "max": beta = np.where(fidx == p1[:, None], beta, 0.0)
if ONLY == "bath": beta = np.where(fidx == p1[:, None], 0.0, beta)
B = np.zeros(N)
for _ in range(K):
    u = rng.random((N, 8))
    newv = np.where(fidx == p1[:, None],
                    ndtri(np.clip(ndtr(v)[:, None] + u * (1 - ndtr(v)[:, None]), 1e-300, 1 - 1e-16)),
                    ndtri(np.clip(u * ndtr(v)[:, None], 1e-300, 1 - 1e-16)))
    dl = d0 + ((newv - y[rows[:, None], fidx]) * beta).sum(1)
    B += (ndtr(np.abs(dl)) - 0.5) / K
nonadj = mass - madj
print(f"{nm} N={N} h={h}: E[mass] A={mass.mean():.5f}  E[madj]={madj.mean():.5f}  B(adj, lin-avg)={B.mean():.5f}  ratio B/madj={B.mean()/madj.mean():.3f}")
print(f"   CV per node: madj {madj.std()/madj.mean():.2f}  B {B.std()/B.mean():.2f}   (se of ratio ~ {np.sqrt((madj.std()/madj.mean())**2/N):.3f})")
print("   |beta| median per coord kind: max", np.median(np.abs(beta[fidx == p1[:, None]])).round(4), " bath", np.median(np.abs(beta[fidx != p1[:, None]])).round(4))

"""T35 step 4 (diagnosis, DEV models only): ARC propagation *conditional on a skeleton* as a proxy for the
conditional accuracy q(S) = P(correct | S), S = positions and values of the top-k inputs (as in E28 stoch_margin).
Given S the other T-k inputs are iid N(0,1) truncated below v_k, so the input cumulants are exact 1-D truncated-normal
cumulants (diagonal tensors); the fixed coordinates have zero variance. Read-out: the cell read-out for p2 (the
position of the 2nd largest, part of S).

Reference: q(S) by MC with M low draws per skeleton (same skeletons). Reports corr(q, q~), Var(q - q~)/Var(q) over
all skeletons and over the worst 1% / 10% (largest 1-q), and the mean bias.
Usage: python diag_skeleton.py T k n_skel M K [dev_model_name]"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "m16")
import numpy as np
from scipy.stats import norm, truncnorm

import arc_kprop as A
import leaderboard as LB

T, k, NS, M, K = (int(a) for a in sys.argv[1:6])
H = 16
dev = LB.models(H, T, dev=True)
mname = sys.argv[6] if len(sys.argv) > 6 else next(iter(dev))
model = dev[mname]


def trunc_cumulants(v, order=4):
    """Cumulants 1..order of N(0,1) | X < v (exact, via the moment recursion)."""
    lam = norm.pdf(v) / norm.cdf(v)
    m = [1.0, -lam]
    for j in range(2, 5):
        m.append((j - 1) * m[j - 2] - v ** (j - 1) * lam)
    k1 = m[1]
    k2 = m[2] - m[1] ** 2
    k3 = m[3] - 3 * m[2] * m[1] + 2 * m[1] ** 3
    k4 = m[4] - 4 * m[3] * m[1] - 3 * m[2] ** 2 + 12 * m[2] * m[1] ** 2 - 6 * m[1] ** 4
    return [k1, k2, k3, k4][:order]


def skeleton_input(pos, vals, order):
    D = H + T
    low = [i for i in range(T) if i not in pos]
    c = trunc_cumulants(vals[-1], order)
    Kin = {d: np.zeros((D,) * d) for d in range(1, order + 1)}
    for p, v in zip(pos, vals):
        Kin[1][H + p] = v
    for i in low:
        for d in range(1, order + 1):
            Kin[d][(H + i,) * d] = c[d - 1]
    return Kin


rng = np.random.default_rng(2026)
q, qt, t_arc = [], [], 0.0
rnn = A.rnn_layers(model)
order = min(4, A.KH.get_d_max(K, A.KH.SIMPLE))
for s in range(NS):
    z = rng.standard_normal(T)
    o = np.argsort(-z)
    pos, vals = [int(i) for i in o[:k]], z[o[:k]]
    # MC reference
    low = [i for i in range(T) if i not in pos]
    x = np.empty((M, T))
    x[:, pos] = vals
    x[:, low] = truncnorm.rvs(-np.inf, vals[-1], size=(M, len(low)), random_state=rng)
    q.append(LB.correct(model, x).mean())
    t0 = time.process_time()
    e = A.propagate(rnn + A.readout_cell(model, pos[1]), skeleton_input(pos, vals, order), K)
    t_arc += time.process_time() - t0
    qt.append(float(e[0]))
q, qt = np.array(q), np.array(qt)
qt_c = np.clip(np.nan_to_num(qt, nan=0.5), 0, 1)
se2 = q * (1 - q) / M            # MC noise in the reference, subtracted from Var(q)


def stats(sel, lab):
    a, b = q[sel], qt_c[sel]
    vq = a.var() - se2[sel].mean()
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
    print(f"{lab:10s} n={sel.sum():4d}  mean q={a.mean():.4f}  mean q~={b.mean():.4f}  corr={r:+.3f}  "
          f"Var(q-q~)/Var(q)={((a - b) ** 2).mean() / max(vq, 1e-12):9.3f}  (Var(q) net of MC noise={vq:.2e})")


print(f"model {mname} T={T} top-{k} skeletons={NS} M={M} K={K}; non-finite q~: {int((~np.isfinite(qt)).sum())}; "
      f"q~ outside [0,1]: {int(((qt < 0) | (qt > 1)).sum())}; ARC CPU {t_arc:.1f}s")
err = 1 - q
stats(np.ones(NS, bool), "all")
stats(err >= np.quantile(err, 0.9), "worst 10%")
stats(err >= np.quantile(err, 0.99), "worst 1%")
np.savez(f"data/arc_cumulants_skel_T{T}_k{k}_K{K}_{mname}.npz", q=q, qt=qt)

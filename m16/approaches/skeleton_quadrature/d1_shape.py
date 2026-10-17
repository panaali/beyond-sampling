"""D1 (pre-registered tests 1, 2, 4; DIAGNOSIS ONLY): where does the variance and error mass of q(S) live?
Skeleton cube coordinates (independent uniforms under the true law, positions uniform over 720 triples):
  U1 = Phi(v1)^10,  U2 = (Phi(v2)/Phi(v1))^9,  U3 = (Phi(v3)/Phi(v2))^8.   The 2-3 tie is U3 -> 1, the 1-2 tie U2 -> 1.
Part A: NS random skeletons, q from M inner draws. Reports
  - variance shares (noise-corrected) of E[q | coordinate bin] for: triple (720 classes), (p2,p3) (90), p2, p3, p1,
    time order of (p1,p2,p3) (6), U1, U2, U3 (20 bins each), d23 = v2 - v3 (20 quantile bins);
  - error-mass tail: share of E[1-q] with d23 > delta, and with d12 > delta.
Part B: L lines; for random (triple, U1, U2), q along a 65-point U3 grid on [0,1] (M2 inner draws), and the
  relative error of Gauss-Legendre rules with k nodes for the line integral of 1-q (reference: the fine grid, Simpson).
usage: python m16/approaches/skeleton_quadrature/d1_shape.py MODEL [NS] [M] [L] [M2]
"""
import itertools
import json
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
M = int(sys.argv[3]) if len(sys.argv) > 3 else 512
L = int(sys.argv[4]) if len(sys.argv) > 4 else 60
M2 = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
model = rv.dev_model(name)
rng = np.random.default_rng(2002)
TRIPLES = np.array(list(itertools.permutations(range(10), 3)))      # 720 ordered (p1,p2,p3)


def vals_from_U(U):
    v1 = norm.ppf(U[:, 0] ** (1 / 10))
    v2 = norm.ppf(U[:, 1] ** (1 / 9) * norm.cdf(v1))
    v3 = norm.ppf(U[:, 2] ** (1 / 8) * norm.cdf(v2))
    return np.stack([v1, v2, v3], 1)


def qhat(pos, val, m):
    free = rv.free_positions(pos)
    out = np.empty(len(pos))
    for a in range(0, len(pos), 100):
        b = slice(a, a + 100)
        low = rv.low_draws(val[b, 2], m, 7, rng)
        x = rv.assemble(pos[b], val[b], free[b], low)
        out[b] = (rv.run(model, x.reshape(-1, rv.T))[0].reshape(-1, m, 10).argmax(2) == pos[b, 1:2]).mean(1)
    return out


# ---- Part A
U = rng.uniform(size=(NS, 3))
tri = rng.integers(0, 720, NS)
pos = TRIPLES[tri]
val = vals_from_U(U)
q = qhat(pos, val, M)
noise = (q * (1 - q)).mean() / (M - 1)
varq = q.var() - noise


def share(keys):
    """noise-corrected variance share of E[q | key] (unbiased within-class correction)."""
    u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    means = np.bincount(inv, q) / cnt
    between = np.sum(cnt * (means - q.mean()) ** 2) / len(q)
    # each class mean carries within-class variance / n_c, correct for it
    within = np.bincount(inv, (q - means[inv]) ** 2) / np.maximum(cnt - 1, 1)
    corr = np.sum(cnt / len(q) * within / cnt)
    return float((between - corr) / varq)


order = np.argsort(pos, 1) @ np.array([1, 3, 9])                      # time-order pattern id
d23, d12 = val[:, 1] - val[:, 2], val[:, 0] - val[:, 1]
bins = lambda z: np.minimum((np.argsort(np.argsort(z)) * 20) // len(z), 19)
out = dict(model=name, NS=NS, M=M, acc=float(q.mean()), var_q=float(varq), Eq1q=float((q * (1 - q)).mean()))
out["shares"] = {k: share(v) for k, v in dict(
    triple=tri, p2p3=pos[:, 1] * 10 + pos[:, 2], p2=pos[:, 1], p3=pos[:, 2], p1=pos[:, 0], order=order,
    U1=bins(U[:, 0]), U2=bins(U[:, 1]), U3=bins(U[:, 2]), d23=bins(d23), d12=bins(d12),
    ).items()}
err = 1 - q
out["err_tail_d23"] = {str(dl): float(err[d23 > dl].sum() / err.sum()) for dl in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)}
out["err_tail_d12"] = {str(dl): float(err[d12 > dl].sum() / err.sum()) for dl in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)}
out["err_share_d12_lt_d23"] = float(err[d12 < d23].sum() / err.sum())
out["top_share"] = {str(f): float(np.sort(err)[::-1][:int(f * NS)].sum() / err.sum()) for f in (0.01, 0.05, 0.1, 0.2)}
np.savez(f"{rv.OUT}/d1A_{name}.npz", U=U, tri=tri, q=q)
print(json.dumps(out, indent=1), flush=True)

# ---- Part B: lines in U3
g = np.linspace(0, 1, 65)
gl = {k: np.polynomial.legendre.leggauss(k) for k in (2, 4, 8, 16)}
rel = {k: [] for k in gl}
lines = []
for i in range(L):
    t = rng.integers(0, 720); u12 = rng.uniform(size=2)
    Ug = np.column_stack([np.full(65, u12[0]), np.full(65, u12[1]), np.clip(g, 1e-9, 1 - 1e-12)])
    ql = qhat(np.repeat(TRIPLES[[t]], 65, 0), vals_from_U(Ug), M2)
    e = 1 - ql
    ref = (e[0:-1:2] + 4 * e[1::2] + e[2::2]).sum() / (3 * 64)
    for k, (xk, wk) in gl.items():
        Uk = np.column_stack([np.full(k, u12[0]), np.full(k, u12[1]), (xk + 1) / 2])
        ek = 1 - qhat(np.repeat(TRIPLES[[t]], k, 0), vals_from_U(Uk), M2)
        rel[k].append((wk @ ek / 2, ref))
    lines.append(e.tolist())
out["lines_U3"] = {}
for k in gl:
    a = np.array(rel[k])
    out["lines_U3"][f"GL{k}"] = dict(sum_abs_err_over_sum_ref=float(np.abs(a[:, 0] - a[:, 1]).sum() / a[:, 1].sum()),
                                     agg_rel_err=float((a[:, 0].sum() - a[:, 1].sum()) / a[:, 1].sum()))
np.save(f"{rv.OUT}/d1B_lines_{name}.npy", np.array(lines))
print(json.dumps(out["lines_U3"], indent=1))
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/d1_{name}.json", "w"), indent=1)

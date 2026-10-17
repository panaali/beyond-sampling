"""D2 (pre-registered test 2; DIAGNOSIS ONLY): can value-rules be shared across the 720 position triples?
For V value points (half from the skeleton law, half from it conditioned on U3 > 0.7, i.e. near the 2-3 tie),
q(triple, v) for ALL 720 triples (M inner draws). Reports, averaged over value points (weighted by the
across-triple variance), the noise-corrected share of the across-triple variance explained by position features,
and the singular-value spectrum of the centred 720 x V matrix (is q(pos, v) ~ sum_r a_r(pos) b_r(v) low rank?).
usage: python m16/approaches/skeleton_quadrature/d2_positions.py MODEL [V] [M]
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402
from d1_shape_lib import vals_from_U  # noqa: E402

name = sys.argv[1]
V = int(sys.argv[2]) if len(sys.argv) > 2 else 24
M = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model = rv.dev_model(name)
rng = np.random.default_rng(3003)
TR = np.array(list(itertools.permutations(range(10), 3)))
U = rng.uniform(size=(V, 3))
U[V // 2:, 2] = 0.7 + 0.3 * U[V // 2:, 2]
vals = vals_from_U(U)
free = rv.free_positions(TR)
Q = np.empty((720, V))
for j in range(V):
    val = np.repeat(vals[[j]], 720, 0)
    for a in range(0, 720, 90):
        b = slice(a, a + 90)
        low = rv.low_draws(val[b, 2], M, 7, rng)
        x = rv.assemble(TR[b], val[b], free[b], low)
        Q[b, j] = (rv.run(model, x.reshape(-1, rv.T))[0].reshape(-1, M, 10).argmax(2) == TR[b, 1:2]).mean(1)
    print(j, flush=True)
noise = Q * (1 - Q) / (M - 1)                                   # per-entry noise variance (unbiased)
feats = dict(p2p3=TR[:, 1] * 10 + TR[:, 2], p1p2=TR[:, 0] * 10 + TR[:, 1], p1p3=TR[:, 0] * 10 + TR[:, 2],
             p2=TR[:, 1], p3=TR[:, 2], p1=TR[:, 0], order=np.argsort(TR, 1) @ np.array([1, 3, 9]),
             last=TR.max(1), order_last=np.argsort(TR, 1) @ np.array([1, 3, 9]) * 10 + TR.max(1),
             first_last=TR.min(1) * 10 + TR.max(1),
             order_p2p3=(np.argsort(TR, 1) @ np.array([1, 3, 9])) * 100 + TR[:, 1] * 10 + TR[:, 2])
tot = np.array([Q[:, j].var() - noise[:, j].mean() for j in range(V)])
out = dict(model=name, V=V, M=M, across_triple_var=tot.tolist(), shares={})
for k, f in feats.items():
    u, inv, cnt = np.unique(f, return_inverse=True, return_counts=True)
    sh = []
    for j in range(V):
        mu = np.bincount(inv, Q[:, j]) / cnt
        between = np.mean((mu[inv] - Q[:, j].mean()) ** 2)
        corr = np.mean(np.bincount(inv, noise[:, j])[inv] / cnt[inv] ** 1)  # noise in class means
        corr = np.sum(np.bincount(inv, noise[:, j]) / cnt) / 720
        sh.append(between - corr)
    out["shares"][k] = dict(n_classes=int(len(u)), weighted_share=float(np.sum(sh) / np.sum(tot)))
C = Q - Q.mean(0)
s = np.linalg.svd(C, compute_uv=False) ** 2
s_noise = noise.sum() * (1 - 1 / 720)
out["svd_energy_cum"] = (np.cumsum(s) / s.sum()).round(4).tolist()[:8]
out["noise_energy_share"] = float(s_noise / s.sum())
np.save(f"{rv.OUT}/d2_Q_{name}.npy", Q)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/d2_{name}.json", "w"), indent=1)

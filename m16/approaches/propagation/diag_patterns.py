# DIAGNOSIS ONLY: how concentrated are (p1,p2,current gate pattern) per step?
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
x = np.random.default_rng(1).standard_normal((500_000, 10)); hs = run(x)
for t in range(10):
    g = np.packbits(hs[:, t + 1] > 0, axis=1)
    o = np.argsort(-x[:, :t + 1], 1)
    p = np.stack([o[:, 0], o[:, min(1, t)]], 1).astype(np.uint8)
    key = np.concatenate([g, p], 1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    cs = np.sort(cnt)[::-1] / len(x)
    n_p = len(np.unique(p, axis=0))
    print(t, 'states', n_p, 'patterns', len(cnt), 'mass top 1x/3x/10x states:', [round(cs[:k * n_p].sum(), 3) for k in (1, 3, 10)])

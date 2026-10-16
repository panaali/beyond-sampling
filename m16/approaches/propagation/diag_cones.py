# DIAGNOSIS ONLY (uses random inputs): how many distinct gate histories / rank paths carry the mass?
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
x = np.random.default_rng(1).standard_normal((1_000_000, 10))
hs = run(x)
g = (hs[:, 1:] > 0).reshape(len(x), -1)
gb = np.packbits(g, axis=1)
o = np.argsort(-x, 1)
# top-2 path: (p1,p2) after each step
path = []
for t in range(1, 11):
    oo = np.argsort(-x[:, :t], 1)
    path.append(oo[:, 0]); path.append(oo[:, 1] if t > 1 else oo[:, 0])
path = np.stack(path, 1).astype(np.uint8)
ok = (hs[:, -1] @ Wo.T).argmax(1) == o[:, 1]
for name, key in [('gates', gb), ('rankpath', path), ('both', np.concatenate([gb, path], 1))]:
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    cs = np.sort(cnt)[::-1]
    print(name, 'distinct', len(cnt), 'mass in top 100/1000/10000:', cs[:100].sum()/len(x), cs[:1000].sum()/len(x), cs[:10000].sum()/len(x),
          'singleton mass', (cnt == 1).sum()/len(x))
    # how pure is correctness within cells
    corr = np.bincount(inv, ok) / cnt
    print('   mass in cells with mixed correctness', (cnt * ((corr > 0) & (corr < 1))).sum() / len(x))

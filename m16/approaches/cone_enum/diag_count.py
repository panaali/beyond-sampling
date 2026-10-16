# DIAGNOSIS ONLY (random inputs): how many error cones (gates, pred, argmax pos, 2nd pos) carry the error mass?
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
N=1_000_000
x = np.random.default_rng(1).standard_normal((N, 10))
hs = run(x)
g = (hs[:, 1:] > 0).reshape(N, -1)
o = np.argsort(-x, 1)
pred = (hs[:, -1] @ Wo.T).argmax(1)
err = pred != o[:, 1]
print('err', err.mean())
key = np.concatenate([np.packbits(g[err],axis=1), np.stack([pred[err], o[err,0], o[err,1]],1).astype(np.uint8)],1)
_, cnt = np.unique(key, axis=0, return_counts=True)
cs = np.sort(cnt)[::-1]; ne=err.sum()
print('distinct error cones', len(cnt), 'singletons frac of err', (cnt==1).sum()/ne)
for k in [10,100,1000,10000,100000]: print(k, cs[:k].sum()/ne)
# all cones (correct) count
_, cnt2 = np.unique(np.packbits(g,axis=1), axis=0, return_counts=True)
print('distinct gate patterns overall', len(cnt2), 'singleton mass', (cnt2==1).sum()/N)

# DIAGNOSIS ONLY: MC check of the enumerated error cones' total mass and the integrator's accuracy.
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '.')
import numpy as np, time, json
from cones import *
from walk import walk
from common import Wi, Wh, Wo, run
n = int(sys.argv[1]); C = Counter(); t0=time.time()
tot, masses, tr = walk((Wi, Wh, Wo), n, C, log_every=n // 10)
keys = walk.keys; idx = {k: i for i, k in enumerate(keys)}
print('walk time', time.time()-t0, 'fwd-equiv', C.fwd_equiv)
hits = np.zeros(len(keys)); N = 0
for s in range(4):
    x = np.random.default_rng(100 + s).standard_normal((1_000_000, 10)); N += len(x)
    hs = run(x); g = np.packbits((hs[:, 1:] > 0).reshape(len(x), -1), axis=1)
    pred = (hs[:, -1] @ Wo.T).argmax(1); o = np.argsort(-x, 1)
    e = np.flatnonzero(pred != o[:, 1])
    for r in e:
        k = (g[r].tobytes(), int(pred[r]), int(o[r, 0]), int(o[r, 1]))
        if k in idx: hits[idx[k]] += 1
mc = hits / N; m = np.array(masses)
print('sum integrator', m.sum(), 'sum MC', mc.sum(), '+-', np.sqrt(hits.sum()) / N)
big = hits >= 20
print('cones with >=20 MC hits', big.sum(), 'median ratio integ/MC', np.median(m[big] / mc[big]) if big.any() else None,
      'IQR', np.percentile(m[big] / mc[big], [25, 75]) if big.any() else None)
json.dump({'n_cones': len(m), 'sum_integ': m.sum(), 'sum_mc': mc.sum(), 'fwd_equiv': C.fwd_equiv, 'lps': C.lps,
           'trace': [(a, float(b), float(c)) for a, b, c in tr]}, open(f'cover_{n}.json', 'w'), indent=1)

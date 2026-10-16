# DIAGNOSIS: validate cone_mass against brute-force MC membership on a few cones of the original model.
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '.')
import numpy as np
from cones import *
from common import Wi, Wh, Wo
model = (Wi, Wh, Wo); C = Counter()
rng = np.random.default_rng(5)
X = rng.standard_normal((2_000_000, 10)).astype(np.float32)
for trial in range(8):
    x = rng.standard_normal(10)
    key, A = cone_of(model, x, C); c, rho = cheb(A, C)
    ms = [cone_mass_mcmc(A, c, C, seed=s)[0] for s in range(8)]
    inside = np.mean([(X[k:k+200000] @ A.T.astype(np.float32) >= 0).all(1).mean() for k in range(0,len(X),200000)])
    print(A.shape, 'qmc %.3e +- %.1e' % (np.mean(ms), np.std(ms)), 'MC %.3e (n=%d hits)' % (inside, inside*len(X)))

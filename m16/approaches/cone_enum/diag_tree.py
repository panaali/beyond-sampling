# DIAGNOSIS ONLY: size of the step-by-step gate-path tree (Hahnloser/Curto pointer): distinct prefixes per step.
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import run
x = np.random.default_rng(7).standard_normal((1_000_000, 10)); hs = run(x)
g = (hs[:, 1:] > 0)
for t in range(1, 11):
    _, c = np.unique(np.packbits(g[:, :t].reshape(len(x), -1), axis=1), axis=0, return_counts=True)
    f1, f2 = (c == 1).sum(), (c == 2).sum()
    print(f't={t:2d} distinct prefixes {len(c):7d}  singleton mass {f1/len(x):.3f}  Chao1 >= {len(c)+f1*f1/(2*max(f2,1)):.3g}')

"""DIAGNOSTIC (sampling): sampled error mass of the original model by relative-gap band (v3 R_EDGES)."""
import sys, numpy as np
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import forward
from common import Wi, Wh, Wo
from v3 import R_EDGES
x = np.random.default_rng(11).standard_normal((1_000_000, 10)); o = np.argsort(x, 1); s = np.sort(x, 1)
out = forward((Wi, Wh, Wo), x).argmax(1); pos = s[:, -2] > 0
for kind, gs, wrong in (('23', (s[:, -2] - s[:, -3]) / s[:, -2], o[:, -3]), ('12', (s[:, -1] - s[:, -2]) / s[:, -1], o[:, -1])):
    print(kind, ' '.join('%.4f' % (pos & (gs >= lo) & (gs < hi) & (out == wrong)).mean() for lo, hi in zip(R_EDGES[:-1], R_EDGES[1:])))

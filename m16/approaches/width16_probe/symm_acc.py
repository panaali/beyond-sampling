"""T216: accuracy change of the symmetrised M_16,10 (prefix.symm_model) vs the target, paired MC on the same 1e7
N(0, I_10) inputs (rng 11; diagnostic): acc each, paired difference with its SE, and prediction disagreement rate.
-> symm_acc.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
import numpy as np
from prefix import target, symm_model
A, B = target(), symm_model(target()); rng = np.random.default_rng(11); n = 0; ca = cb = 0; d = []; dis = 0


def pred(m, x):
    Wi, Wh, Wo = m; h = np.zeros((len(x), len(Wi)))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


for _ in range(20):
    x = rng.standard_normal((500_000, 10)); y = np.argsort(x, 1)[:, -2]; pa, pb = pred(A, x), pred(B, x)
    a, b = pa == y, pb == y; ca += a.sum(); cb += b.sum(); d.append(b.astype(float) - a); dis += (pa != pb).sum(); n += len(x)
d = np.concatenate(d)
r = dict(n=n, acc_target=ca / n, acc_sym=cb / n, diff_pp=100 * d.mean(), diff_se_pp=100 * d.std() / np.sqrt(n), disagree=dis / n,
         max_rel_weight_change=float(max(np.linalg.norm(np.c_[B[0], B[1]][j] - np.c_[A[0], A[1]][j]) / np.linalg.norm(np.c_[A[0], A[1]][j]) for j in (1, 4, 6, 7, 8))))
json.dump(r, open(os.path.join(HERE, "symm_acc.json"), "w"), indent=1); print(r)

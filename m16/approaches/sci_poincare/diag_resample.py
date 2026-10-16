"""Diagnostic (uses random inputs): which inputs does correctness depend on?
Keep the top-k values and their positions, resample the remaining inputs from N(0,1) conditioned to stay below
the k-th largest. Report how often correctness changes (vs the model's error rate)."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from arc_eval import forward, Wi0, Wh0, Wo0
from scipy.special import ndtr, ndtri
rng = np.random.default_rng(1)
M = (Wi0, Wh0, Wo0)
N = 400_000
x = rng.standard_normal((N, 10))
tgt = np.argsort(x, 1)[:, -2]
ok = forward(M, x).argmax(1) == tgt
print("acc", ok.mean())
order = np.argsort(-x, 1)
for k in (2, 3, 4, 5, 6, 8):
    xr = x.copy()
    thr = np.take_along_axis(x, order[:, k-1:k], 1)  # k-th largest
    low = order[:, k:]
    u = rng.random(low.shape) * ndtr(thr)
    np.put_along_axis(xr, low, ndtri(u), 1)
    ok2 = forward(M, xr).argmax(1) == tgt
    print(f"k={k} flip rate {np.mean(ok != ok2):.4f}  acc after {ok2.mean():.4f}  P(err both) {np.mean(~ok & ~ok2):.4f}")

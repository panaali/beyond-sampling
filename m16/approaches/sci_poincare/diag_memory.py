"""Diagnostic: forgetting. Keep top-3 (values+positions) and all inputs at times >= 10-k;
resample the other early inputs below m3 (conditioned). Flip rate of correctness vs k."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from arc_eval import forward, Wi0, Wh0, Wo0
from scipy.special import ndtr, ndtri
rng = np.random.default_rng(2)
M = (Wi0, Wh0, Wo0); N = 300_000
x = rng.standard_normal((N, 10)); tgt = np.argsort(x, 1)[:, -2]
ok = forward(M, x).argmax(1) == tgt
order = np.argsort(-x, 1); m3 = np.take_along_axis(x, order[:, 2:3], 1)
top3 = np.zeros_like(x, bool); np.put_along_axis(top3, order[:, :3], True, 1)
for k in (0, 1, 2, 3, 4, 5, 6, 7):
    mask = ~top3.copy(); mask[:, 10-k:] = False   # resample these
    xr = x.copy(); u = rng.random(x.shape) * ndtr(m3); xr[mask] = ndtri(u)[mask]
    ok2 = forward(M, xr).argmax(1) == tgt
    print(f"keep last {k}: flip {np.mean(ok!=ok2):.4f}")
# and the converse: resample only the last k non-top3
for k in (1, 2, 3):
    mask = ~top3.copy(); mask[:, :10-k] = False
    xr = x.copy(); u = rng.random(x.shape) * ndtr(m3); xr[mask] = ndtri(u)[mask]
    ok2 = forward(M, xr).argmax(1) == tgt
    print(f"resample only last {k}: flip {np.mean(ok!=ok2):.4f}")

"""Ray-search design points on a chamber subset; saved for comparison with sampled chamber truth."""
import sys, time, numpy as np, torch
torch.set_num_threads(1)
from form import *
from common import Wi, Wh, Wo
tm = tmodel((Wi, Wh, Wo))
perms = torch.as_tensor(chambers(1, seed=0))
idx = np.arange(0, 720, 6)          # 120 chambers
out = {}
for k in (0, 2, 3):
    t = time.time(); C.fwd = 0
    b, a, u = design_points(tm, perms[idx], torch.full((len(idx),), k), iters=int(sys.argv[1]))
    out[f'b{k}'] = b; out[f'a{k}'] = a; out[f'u{k}'] = u
    print(k, round(time.time()-t, 1), 'evals/problem', C.fwd / len(idx), np.quantile(b, [0, .1, .5, .9, 1]).round(2), flush=True)
np.savez(f'diag2_{sys.argv[1]}.npz', idx=idx, **out)

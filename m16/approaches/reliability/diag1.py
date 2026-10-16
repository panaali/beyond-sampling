"""Diagnosis: FORM per-mode probability vs sampled per-chamber mode probability (original model)."""
import time, numpy as np, torch
from scipy.stats import norm
from form import *
from common import Wi, Wh, Wo
tm = tmodel((Wi, Wh, Wo))
perms = torch.as_tensor(chambers(1, seed=0))
Cn = len(perms)
res = {}
for k in (0, 2, 3):
    t = time.time(); C.fwd = 0
    b, a, u, g = hlrf(tm, perms, torch.full((Cn,), k), iters=25)
    res[k] = (b, a)
    print(k, 'time', round(time.time()-t,1), 'evals', C.fwd, 'final |g| med', np.median(np.abs(g)), 'beta quantiles', np.quantile(b,[0,.1,.5,.9,1]).round(2))
np.savez('diag1_form.npz', **{f'b{k}': res[k][0] for k in res}, **{f'a{k}': res[k][1] for k in res})
# truth per chamber by sampling (diagnosis only)
M = 2000
rng = torch.Generator().manual_seed(1)
fails = {k: np.zeros(Cn) for k in (0, 2, 3, 'other', 'any')}
with torch.no_grad():
    for c0 in range(0, Cn, 60):
        P = perms[c0:c0+60].repeat_interleave(M, 0)
        u = torch.randn(len(P), 10, generator=rng)
        x = place(sorted_from_u(u), P)
        pred = logits(tm, x).argmax(1)
        rank = (P == pred[:, None]).long().argmax(1)
        rank = rank.view(-1, M)
        for k in (0, 2, 3):
            fails[k][c0:c0+60] = (rank == k).double().mean(1).numpy()
        fails['any'][c0:c0+60] = (rank != 1).double().mean(1).numpy()
for k in (0, 2, 3):
    pf = norm.cdf(-res[k][0])
    print('rank', k, 'FORM mean', pf.mean().round(5), 'MC mean', fails[k].mean().round(5), 'corr', np.corrcoef(pf, fails[k])[0,1].round(3))
print('any MC', fails['any'].mean())
np.savez('diag1_mc.npz', **{str(k): v for k, v in fails.items()})

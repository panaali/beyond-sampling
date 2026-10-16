"""Multi-start design points for the rank-1 (picks max) mode on one model: are there several design points?"""
import sys, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '/home/user/clean-neurons/m16')
from scipy.stats import norm
from arc_eval import noisy_models, Wi0, Wh0, Wo0
from form import *
models = {'orig': (Wi0, Wh0, Wo0)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s, seed=0)):
        models[f's{s}_{k}'] = m
name, k = sys.argv[1], int(sys.argv[2]); tm = tmodel(models[name])
perms = torch.as_tensor(chambers(1, seed=0)); Cn = 720; ks = torch.full((Cn,), k)
G = np.load(f'mcg_{name}.npy')[:, k]
starts = {'grad': None}
for j in range(10):
    for sg in (1, -1):
        e = torch.zeros(Cn, 10); e[:, j] = sg; starts[f'{"+" if sg>0 else "-"}e{j}'] = e
B = {}; A = {}
for nm, d0 in starts.items():
    b, a, u = design_points(tm, perms, ks, True, iters=6, d0=d0)
    B[nm] = b; A[nm] = a
    print(nm, 'mean Phi(-b)', norm.cdf(-b).mean().round(5), flush=True)
Bm = np.stack(list(B.values())); Am = np.stack(list(A.values()))
best = Bm.min(0)
print('MC', G.mean().round(5), 'grad-start', norm.cdf(-B['grad']).mean().round(5), 'best-of-starts', norm.cdf(-best).mean().round(5))
# distinct design points: union of linearised half-spaces from all starts with beta within 1.5 of best
np.savez(f'multi_{name}_{k}.npz', B=Bm, A=Am)

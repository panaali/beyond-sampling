import sys, numpy as np, torch
from scipy.stats import norm
from form import chambers, tmodel, C
from sorm import *
from common import Wi, Wh, Wo
tm = tmodel((Wi, Wh, Wo)); modes = (0, 2, 3, 4, 5, 6, 7, 8, 9)
for pos in (1, 0):
    d = np.load(f'orig_modes_{pos}.npz'); b, u = d['b'], d['u']
    R = np.load(f'diag1_mc_1_{pos}.npy')
    perms = torch.as_tensor(chambers(1, seed=0)); Cn = len(perms)
    P = perms.repeat(len(modes), 1); ks = torch.as_tensor(np.repeat(modes, Cn))
    U = torch.as_tensor(u.reshape(-1, 10))
    w, g = x_grad(tm, U, P, ks, bool(pos))
    H, gu = u_hessian(U, w, bool(pos))
    kap = curvatures(H, gu)
    np.save(f'orig_kappa_{pos}.npy', kap.reshape(len(modes), Cn, 9))
    for kind in ('breitung', 'hr'):
        pf = sorm_pf(b.ravel(), kap, kind).reshape(len(modes), Cn)
        print(pos, kind, ' '.join(f'r{k+1}:{pf[i].mean():.5f}/{R[:, k].mean():.5f}' for i, k in enumerate(modes)))
    print(pos, 'FORM ', ' '.join(f'r{k+1}:{norm.cdf(-b[i]).mean():.5f}' for i, k in enumerate(modes)))
    print('kappa quantiles', np.quantile(kap, [.01, .1, .5, .9, .99]).round(3))

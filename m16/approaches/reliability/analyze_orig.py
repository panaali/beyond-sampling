import numpy as np
from scipy.stats import norm
from system import combine
from rosen import G0
modes = (0, 2, 3, 4, 5, 6, 7, 8, 9)
for pos in (1, 0):
    d = np.load(f'orig_modes_{pos}.npz'); b, a = d['b'], d['a']
    R = np.load(f'diag1_mc_1_{pos}.npy')
    pf = norm.cdf(-b)
    print('region pos' if pos else 'region blind')
    for i, k in enumerate(modes):
        print(f'  rank{k+1}: FORM {pf[i].mean():.5f}  MC {R[:, k].mean():.5f}  corr {np.corrcoef(pf[i], R[:, k])[0,1]:.2f}  notfound {np.mean(np.abs(b[i])>=8):.2f}')
    cmb = combine(b, a)
    print('  fail: sum', cmb['sum'].mean().round(5), 'ditlevsen', cmb['lo'].mean().round(5), cmb['up'].mean().round(5), 'MC', (1 - R[:, 1]).mean().round(5))

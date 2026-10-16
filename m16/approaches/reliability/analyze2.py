import numpy as np
from scipy.stats import norm
from system import combine, hr_mvn
from sorm import sorm_pf
from rosen import G0
modes = (0, 2, 3, 4, 5, 6, 7, 8, 9)
acc = {}
for pos in (1, 0):
    d = np.load(f'orig_modes_{pos}.npz'); b, a = d['b'], d['a']
    kap = np.load(f'orig_kappa_{pos}.npy')
    R = np.load(f'diag1_mc_1_{pos}.npy')
    pf = sorm_pf(b.ravel(), kap.reshape(-1, 9), 'hr').reshape(b.shape)
    beq = -norm.ppf(np.clip(pf, 1e-15, 1 - 1e-15))
    for name, bb in (('FORM', b), ('SORM', beq)):
        cmb = combine(bb, a)
        hr = 1 - hr_mvn(bb, a)
        print(pos, name, 'sum', cmb['sum'].mean().round(5), 'Ditlevsen', cmb['lo'].mean().round(5), cmb['up'].mean().round(5),
              'HR-MVN', hr.mean().round(5), 'MC', (1 - R[:, 1]).mean().round(5), 'corr', np.corrcoef(hr, 1 - R[:, 1])[0, 1].round(2))
        acc[(pos, name)] = (cmb['lo'].mean(), cmb['up'].mean(), hr.mean())
for name in ('FORM', 'SORM'):
    for j, lab in enumerate(('Dit-lo', 'Dit-up', 'HR')):
        f = (1 - G0) * acc[(1, name)][j] + G0 * acc[(0, name)][j]
        print(name, lab, 'accuracy', round(1 - f, 5), 'truth 0.95318')

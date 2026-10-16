import sys; sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '.')
import numpy as np
from cones import *
from common import Wi, Wh, Wo
model = (Wi, Wh, Wo); C = Counter()
rng = np.random.default_rng(5)
X = rng.standard_normal((2_000_000, 10)).astype(np.float32)
x = rng.standard_normal(10)
key, A = cone_of(model, x, C); c, rho = cheb(A, C)
ins = np.concatenate([X[k:k+200000][(X[k:k+200000] @ A.T.astype(np.float32) >= 0).all(1)] for k in range(0,len(X),200000)]).astype(float)
print(len(ins), 'MC mass', len(ins)/len(X))
mi = ins.mean(0); print('MC mean dir', np.round(mi/np.linalg.norm(mi),2), 'norm', np.linalg.norm(mi))
for steps,K in [((30,30,40),8),((40,40,40,40),8),((60,60,60,60),16)]:
    f0=C.flops; est, rse, (Y, Lb, b, w, cm) = cone_mass_ahr(A, c, C, K=K, phases=steps); print('fwd-equiv', (C.flops-f0)/FWD)
    print(steps, K, 'est', est, 'rse', rse, 'cos(cm, MCmean)', cm@mi/np.linalg.norm(mi))
# oracle: use MC-inside points for fit
cm = mi/np.linalg.norm(mi); d=10
Q = np.linalg.svd(np.eye(d) - np.outer(cm, cm))[0][:, :d - 1]
Yz = (ins - np.outer(ins @ cm, cm)) @ Q
mu = Yz.mean(0); Sig = np.cov(Yz.T)*1.5
Lc=np.linalg.cholesky(Sig); Z0=rng.standard_normal((4096,9)); Z=mu+Z0@Lc.T
ac=A@cm; R=-(Z@(A@Q).T)/ac; Lb=R.max(1)
lw = -0.5*(Z**2).sum(1)+0.5*(Z0**2).sum(1)+np.log(np.diag(Lc)).sum()+log_ndtr(-Lb)
print('oracle-fit IS', np.exp(lw).mean(), np.exp(lw).std()/64)

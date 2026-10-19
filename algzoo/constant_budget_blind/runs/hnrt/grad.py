import numpy as np, probe
rng=np.random.default_rng(2)
N=20000
x=rng.standard_normal((N,10))
H=probe.hidden(x)
G=np.zeros((N,10,16)); e=1e-2
for j in range(10):
    xp=x.copy(); xp[:,j]+=e
    G[:,j,:]=(probe.hidden(xp)-H)/e
np.savez('grad.npz',x=x,H=H,G=G)
# homogeneity check
print("homog err",np.abs((G*x[:,:,None]).sum(1)-H).max())

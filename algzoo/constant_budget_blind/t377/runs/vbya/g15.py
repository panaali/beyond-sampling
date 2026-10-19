import numpy as np, probe
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(21)
x=rng.standard_normal((20000,10)); H=probe.hidden(x)
rk=np.argsort(np.argsort(-x[:,:9],1),1)  # rank within x0..x8
eps=1e-3
for i in [0,12,29,31]:
    print('neuron',i,' mean dh/dx_t by rank of x_t within x0..x8 (rows=t, cols=rank0..8)')
    M=np.zeros((10,9))
    for t in range(10):
        y=x.copy(); y[:,t]+=eps; d=(probe.hidden(y)[:,i]-H[:,i])/eps
        if t<9: M[t]=[d[rk[:,t]==k].mean() for k in range(9)]
        else: M[t]=[d.mean()]+[0]*8
    print(M)

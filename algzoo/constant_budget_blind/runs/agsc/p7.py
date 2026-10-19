import numpy as np
from traj import traj; from feat import r2
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(6); N=20000
x=rng.standard_normal((N,10)); H=traj(x); np.save('H6.npy',H); np.save('x6.npy',x)
print('mean h_t per neuron'); print(H.mean(0)[1:].T)
print('frac active'); print((H[:,1:]>1e-6).mean(0).T)
for t in range(2,11):
    p=x[:,:t]; s=np.sort(p,1)[:,::-1]
    A=np.c_[s[:,0],s[:,1],x[:,t-1],np.ones(N)]
    out=[]
    for i in range(8):
        y=H[:,t,i]; c=np.linalg.lstsq(A,y,rcond=None)[0]; out.append(round(r2(y,A@c),2))
    print(t,out)

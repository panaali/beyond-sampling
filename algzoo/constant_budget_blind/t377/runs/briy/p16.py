import numpy as np, probe
from dic3 import relufit
x=np.random.default_rng(7).standard_normal((30000,10))
np.set_printoptions(linewidth=220,suppress=True,precision=1)
H={}
for t in range(1,11):
    z=np.zeros_like(x); z[:,10-t:]=x[:,:t]; H[t]=probe.hidden(z)
np.save('x7.npy',x); np.save('Ht.npy',np.stack([H[t] for t in range(1,11)],0))
for i in [15,12,3]:
  print('neuron',i)
  for t in range(1,11):
    y=H[t][:,i]; xs=x[:,:t]; M=xs.max(1)
    c,r=relufit(np.c_[xs,M],y)
    print(' t',t,'frac0 %.2f'%(y<=1e-6).mean(),'R2 %.3f'%r,c)

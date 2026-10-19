import numpy as np
from dic3 import relufit, pref
x=np.load('x7.npy'); Ht=np.load('Ht.npy')  # Ht[t-1]= state after reading x0..x_{t-1}
np.set_printoptions(linewidth=220,suppress=True,precision=2)
for t in range(1,11):
    xs=x[:,:t]; y=Ht[t-1][:,14]; am=xs.argmax(1); M=xs.max(1)
    print('t',t,'frac active %.3f'%(y>1e-6).mean(), 'act by am', np.round([ (y[am==k]>1e-6).mean() for k in range(t)],2), 'mean by am',np.round([ y[am==k].mean() for k in range(t)],2))
h=Ht[9]; s=np.sort(x,1)[:,::-1]; am=x.argmax(1); g=(am==6)
M5=x[:,:6].max(1)
for F in [[x[:,6],s[:,1],x[:,9]],[x[:,6],M5,x[:,7],x[:,8],x[:,9]],[x[:,6],s[:,1],x[:,7],x[:,8],x[:,9]]]:
    B=np.stack(F,1)*g[:,None]; c,r=relufit(B,h[:,14]); print('R2 %.3f'%r,c)

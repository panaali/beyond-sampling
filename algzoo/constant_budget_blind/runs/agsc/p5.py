import numpy as np, probe
from feat import *; from fit import *
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(4)
N=100000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1)
r=np.argsort(o,1)  # rank of each position
E1=np.zeros((N,10));E1[np.arange(N),o[:,0]]=s[:,0]
E2=np.zeros((N,10));E2[np.arange(N),o[:,1]]=s[:,1]
R=x*(r>=2)  # other values by position
A=np.c_[E1,E2,R,np.ones(N)]
for i in range(8):
    y=h[:,i]; c=np.linalg.lstsq(A,y,rcond=None)[0]
    print(i,'R2',round(r2(y,A@c),3),'const',round(c[-1],2)); print('  max@t',c[:10]); print('  sec@t',c[10:20]); print('  oth@t',c[20:30])

import numpy as np, sys
from lib import relufit
X=np.load('x0.npy');H=np.load('h0.npy')
x=X[:15000];h=H[:15000];xt=X[80000:100000];ht=H[80000:100000];one=np.ones(len(x))
def mS(x,S): return x[:,S].max(1)
subs=[[k for k in range(10) if m>>k&1] for m in range(1,1023)]
def ls(A,y,g):
    c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None); return 1-((A[g]@c-y[g])**2).mean()/y[g].var()
for i in [int(a) for a in sys.argv[1].split(',')]:
    y=h[:,i];g=y>0.3; res=[]
    for S in subs:
        C=[k for k in range(10) if k not in S]
        res.append((ls(np.c_[mS(x,S),mS(x,C),one],y,g),S))
    res.sort(reverse=True)
    for r,S in res[:3]:
        C=[k for k in range(10) if k not in S]
        c,_=relufit(np.c_[mS(x,S),mS(x,C),one],y)
        pt=np.maximum(np.c_[mS(xt,S),mS(xt,C),np.ones(len(xt))]@c,0)
        print(i,'S=',S,np.round(c,2),'R2test',round(1-((pt-ht[:,i])**2).mean()/ht[:,i].var(),3),flush=True)

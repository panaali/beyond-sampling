import numpy as np, sys
from lib import relufit
X=np.load('x0.npy');H=np.load('h0.npy')
x=X[:12000];h=H[:12000];xt=X[80000:100000];ht=H[80000:100000]
subs=[[k for k in range(10) if m>>k&1] for m in range(1,1024)]
V=np.stack([x[:,S].max(1) for S in subs],1); Vt=np.stack([xt[:,S].max(1) for S in subs],1)
one=np.ones(len(x))
def ls(A,y,g):
    c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None); return 1-((A[g]@c-y[g])**2).mean()/y[g].var()
for i in [int(a) for a in sys.argv[1].split(',')]:
    y=h[:,i]; g=y>0.3; sel=[]
    for step in range(4):
        s,j=max((ls(np.c_[V[:,sel+[j]],one],y,g),j) for j in range(V.shape[1]) if j not in sel); sel.append(j)
        c,_=relufit(np.c_[V[:,sel],one],y); pt=np.maximum(np.c_[Vt[:,sel],np.ones(len(xt))]@c,0)
        print(i,[subs[k] for k in sel],np.round(c,2),'R2test',round(1-((pt-ht[:,i])**2).mean()/ht[:,i].var(),3),flush=True)

import numpy as np, sys
from lib import relufit
X=np.load('x0.npy');H=np.load('h0.npy')
def feats(x):
    names=[];cols=[]
    for a in range(10):
        for b in range(a,10):
            names.append(f"max{a}-{b}");cols.append(x[:,a:b+1].max(1))
    for a in range(10):
        for b in range(a+1,10):
            names.append(f"sec{a}-{b}");cols.append(np.sort(x[:,a:b+1],1)[:,-2])
    return names,np.stack(cols,1)
names,V=feats(X[:15000]);_,Vt=feats(X[80000:100000]);h=H[:15000];ht=H[80000:100000]
one=np.ones(15000)
def ls(A,y,g):
    c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None); return 1-((A[g]@c-y[g])**2).mean()/y[g].var()
K=V.shape[1]
for i in [int(a) for a in sys.argv[1].split(',')]:
    y=h[:,i]; g=y>0.3
    sc=np.array([ls(np.c_[V[:,j],one],y,g) for j in range(K)]); top=list(np.argsort(-sc)[:25])
    best=max((ls(np.c_[V[:,j],V[:,k],one],y,g),[j,k]) for j in top for k in range(K) if k!=j)
    sel=best[1]
    for step in range(3):
        A=np.c_[V[:,sel],one]; c,r=relufit(A,y); At=np.c_[Vt[:,sel],np.ones(len(Vt))]
        rt=1-((np.maximum(At@c,0)-ht[:,i])**2).mean()/ht[:,i].var()
        print(i,[names[j] for j in sel],np.round(c,2),'R2test',round(rt,4),flush=True)
        if step==2: break
        s,j=max((ls(np.c_[V[:,sel+[j]],one],y,g),j) for j in range(K) if j not in sel); sel=sel+[j]

import numpy as np
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
o=np.argsort(-x,1);am=o[:,0];M=x.max(1);S=-np.sort(-x,1)[:,1]
ev=(am%2==0).astype(float)
pm=np.maximum.accumulate(x,1)
cap=np.c_[x[:,:1],np.minimum(x[:,1:],pm[:,:-1])]
def fit(X,y):
    c,*_=np.linalg.lstsq(X,y,rcond=None);return 1-((y-X@c)**2).mean()/y.var(),c
fams={'F1':(np.c_[pm[:,7],pm[:,8],ev*M],x),'F2':(np.c_[pm[:,7],pm[:,8],ev*M],cap),'F3':(np.c_[pm[:,8],M,ev*M],x),'F4':(np.c_[pm[:,7],pm[:,8],M],x)}
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in [0,1,2,4,7,9,10,11,13,15]:
    y=H[:,i];out=[]
    for fn,(B,Z) in fams.items():
        P=[];best=(-1,None,None)
        for it in range(6):
            cand=best
            for t in range(10):
                if t in P: continue
                Q=P+[t];rest=[u for u in range(10) if u not in Q]
                r,c=fit(np.c_[B,Z[:,Q].sum(1),Z[:,rest].sum(1)],y)
                if r>cand[0]+1e-4: cand=(r,Q,c)
            if cand[1] is None or cand[1]==P: break
            best=cand;P=cand[1]
        out.append(f"{fn} {best[0]:.3f} P={sorted(best[1])} c={np.round(best[2],1)}")
    print(i,"\n   ".join(out))

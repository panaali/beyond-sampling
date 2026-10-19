import numpy as np
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
o=np.argsort(-x,1);am=o[:,0];M=x.max(1);S=-np.sort(-x,1)[:,1]
ev=(am%2==0).astype(float)
pm=np.maximum.accumulate(x,1)
def fit(X,y):
    c,*_=np.linalg.lstsq(X,y,rcond=None);return 1-((y-X@c)**2).mean()/y.var(),c
fams={'G1':(np.c_[pm[:,7],pm[:,8],ev*M],True),'G2':(np.c_[pm[:,7],pm[:,8],S,ev*M],False),'G3':(np.c_[pm[:,7],pm[:,8],S],True),'G4':(np.c_[pm[:,8],S,ev*M],True),'G5':(np.c_[pm[:,7],pm[:,8],ev*M,S],True)}
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in [0,1,2,4,7,9,10,11,13,15]:
    y=H[:,i];out=[]
    for fn,(B,rest_on) in fams.items():
        P=[];best=(-1,None,None)
        for it in range(6):
            cand=best
            for t in range(10):
                if t in P: continue
                Q=P+[t];rest=[u for u in range(10) if u not in Q]
                X=np.c_[B,x[:,Q].sum(1)]
                if rest_on: X=np.c_[X,x[:,rest].sum(1)]
                r,c=fit(X,y)
                if r>cand[0]+1e-4: cand=(r,Q,c)
            if cand[1] is None or cand[1]==P: break
            best=cand;P=cand[1]
        out.append(f"{fn} {best[0]:.3f} P={sorted(best[1])}")
    print(i," | ".join(out))

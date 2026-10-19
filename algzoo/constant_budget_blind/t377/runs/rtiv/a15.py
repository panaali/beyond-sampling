import numpy as np, sys
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
o=np.argsort(-x,1);am=o[:,0];M=x.max(1)
pm=np.maximum.accumulate(x,1)
OH=(np.arange(10)[None]==am[:,None])
def fit(X,y):
    c,*_=np.linalg.lstsq(X,y,rcond=None);return 1-((y-X@c)**2).mean()/y.var(),c
def design(Q,P):
    rest=[u for u in range(10) if u not in P]
    return np.c_[pm[:,7],pm[:,8],M*OH[:,Q].any(1),x[:,P].sum(1),x[:,rest].sum(1)]
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in [int(a) for a in sys.argv[1:]]:
    y=H[:,i];Q=[];P=[];best=(fit(design(Q,P),y)[0],None)
    for it in range(12):
        cand=None
        for t in range(10):
            for which in 'QP':
                Q2=Q+[t] if which=='Q' and t not in Q else Q
                P2=P+[t] if which=='P' and t not in P else P
                if (Q2,P2)==(Q,P): continue
                r,c=fit(design(Q2,P2),y)
                if cand is None or r>cand[0]: cand=(r,Q2,P2,c)
        if cand[0]<best[0]+0.002: break
        best=cand;Q,P=cand[1],cand[2]
    print(i,"%.3f"%best[0],"Q",sorted(Q),"P",sorted(P),best[3])

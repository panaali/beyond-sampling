import numpy as np, sys
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
o=np.argsort(-x,1);am=o[:,0];M=x.max(1);S=-np.sort(-x,1)[:,1]
pm=np.maximum.accumulate(x,1)
OH=(np.arange(10)[None]==am[:,None])
def fit(X,y):
    c,*_=np.linalg.lstsq(X,y,rcond=None);return 1-((y-X@c)**2).mean()/y.var(),c
bases={'Hs':lambda: [pm[:,8],S],'Hm':lambda: [M,S],'H78':lambda: [pm[:,7],pm[:,8]]}
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in [int(a) for a in sys.argv[1:]]:
  y=H[:,i]
  for bn,bf in bases.items():
    B=bf()
    def design(Q,P):
        rest=[u for u in range(10) if u not in P]
        return np.c_[B[0],B[1],M*OH[:,Q].any(1),x[:,P].sum(1),x[:,rest].sum(1)]
    Q=[];P=[];best=(fit(design(Q,P),y)[0],None)
    for it in range(14):
        cand=None
        for t in range(10):
            for which in 'QP':
                Q2=sorted(set(Q+[t])) if which=='Q' else Q
                P2=sorted(set(P+[t])) if which=='P' else P
                if (Q2,P2)==(Q,P): continue
                r,c=fit(design(Q2,P2),y)
                if cand is None or r>cand[0]: cand=(r,Q2,P2,c)
        if cand[0]<best[0]+0.002: break
        best=cand;Q,P=cand[1],cand[2]
    print(i,bn,"%.3f"%best[0],"Q",Q,"P",P,best[3])

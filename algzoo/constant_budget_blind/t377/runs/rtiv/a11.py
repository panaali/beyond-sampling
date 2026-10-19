import numpy as np, itertools
d=np.load('d1.npz');x=d['x'][:50000];H=d['H'][:50000]
o=np.argsort(-x,1);am=o[:,0];M=x.max(1);S=-np.sort(-x,1)[:,1]
ev=(am%2==0).astype(float)
def fit(X,y):
    c,*_=np.linalg.lstsq(X,y,rcond=None);return 1-((y-X@c)**2).mean()/y.var(),c
def greedyP(base,y,shared):
    # choose position set P; shared: one coef on sum_P x, else separate coefs up to 3
    best=(fit(base,y)[0],[],None)
    if shared:
        P=[]
        while True:
            cand=None
            for t in range(10):
                if t in P: continue
                Q=P+[t];r,c=fit(np.c_[base,x[:,Q].sum(1)],y)
                if cand is None or r>cand[0]: cand=(r,Q,c)
            if cand[0]>best[0]+0.002: best=cand;P=cand[1]
            else: break
    else:
        P=[]
        for k in range(3):
            cand=None
            for t in range(10):
                if t in P: continue
                Q=P+[t];r,c=fit(np.c_[base,x[:,Q]],y)
                if cand is None or r>cand[0]: cand=(r,Q,c)
            best=cand;P=cand[1]
    return best
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in [0,1,2,4,7,9,10,11,13,15]:
    y=H[:,i]
    r1=greedyP(np.c_[M,S],y,False)
    r2=greedyP(np.c_[ev*M,ev*S,(1-ev)*M,(1-ev)*S],y,True)
    r3=greedyP(np.c_[M,S],y,True)
    print(i,"V1 %.3f"%r1[0],r1[1],r1[2]," V2 %.3f"%r2[0],sorted(r2[1]),r2[2]," V3(M,S,sharedP) %.3f"%r3[0],sorted(r3[1]),r3[2])

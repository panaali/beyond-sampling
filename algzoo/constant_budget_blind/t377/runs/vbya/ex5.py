import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
x=np.load('x4.npy'); G=np.load('G4.npy'); pre=x[:,:9]
sets={3:[1,4,7],18:[0,5,7,8],20:[0,3,4],22:[3,6,7,8],28:[0,1,4,5,6,8],30:[2,4,6,7,8],17:[0,1,2],9:[0,5,6,7,8],7:[0,1,2,3,4,5,6,7]}
def fit(g,F):
    A=np.c_[F,np.ones(len(g))]; c,*_=np.linalg.lstsq(A,g,rcond=None); r=g-A@c; return c,1-r.var()/g.var()
for i,S in sets.items():
    N=[t for t in range(9) if t not in S]
    mS=pre[:,S].max(1); mN=pre[:,N].max(1)
    c,r2=fit(G[:,i],np.c_[mS,mN]); print(i,S,'g~mS,mN',c,'R2 %.3f'%r2)
    # greedy search best set: try toggling
    best=(r2,list(S))
    improved=True
    while improved:
        improved=False
        for t in range(9):
            S2=[u for u in best[1] if u!=t] if t in best[1] else sorted(best[1]+[t])
            N2=[u for u in range(9) if u not in S2]
            if not S2 or not N2: continue
            c2,r=fit(G[:,i],np.c_[pre[:,S2].max(1),pre[:,N2].max(1)])
            if r>best[0]+1e-4: best=(r,S2); improved=True
    S2=best[1]; N2=[u for u in range(9) if u not in S2]
    c2,r=fit(G[:,i],np.c_[pre[:,S2].max(1),pre[:,N2].max(1)])
    print('   best set',S2,c2,'R2 %.3f'%r)

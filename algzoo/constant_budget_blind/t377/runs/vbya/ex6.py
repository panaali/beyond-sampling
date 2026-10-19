import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
x=np.load('x4.npy'); G=np.load('G4.npy'); pre=x[:,:9]
sets={3:[1,4,7],18:[0,5,7,8],20:[0,3,4],22:[3,6,7,8],28:[0,1,4,5,6,8],30:[2,4,6,7,8],17:[0,1,2],9:[0,5,6,7,8]}
def fit(g,F):
    A=np.c_[F,np.ones(len(g))]; c,*_=np.linalg.lstsq(A,g,rcond=None); r=g-A@c; return c,1-r.var()/g.var()
def r2of(g,S,d):
    b=np.zeros(9); b[S]=d
    m=(pre+b).max(1)
    return fit(g,m[:,None])
for i,S in sets.items():
    g=G[:,i]
    best=None
    for it in range(3):
        for d in np.arange(-2,2.01,0.1):
            c,r=r2of(g,S,d)
            if best is None or r>best[0]: best=(r,d,list(S),c)
        d=best[1]; S=best[2]
        for t in range(9):
            S2=[u for u in S if u!=t] if t in S else sorted(S+[t])
            c,r=r2of(g,S2,d)
            if r>best[0]: best=(r,d,S2,c); S=S2
    print(i,'R2 %.3f d=%.2f S=%s coef'%(best[0],best[1],best[2]),best[3])
    # residual dependence on per-position x
    b=np.zeros(9); b[best[2]]=best[1]; m=(pre+b).max(1)
    c,r=fit(g,np.c_[m,pre]); print('    +x lin R2 %.3f'%r,c)

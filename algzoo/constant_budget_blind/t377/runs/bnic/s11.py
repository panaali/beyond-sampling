import numpy as np, itertools
x=np.load('x1.npy')[:50000]; h=np.load('h1.npy')[:50000]
M=np.maximum.accumulate(x,1); m=M[:,9]; s=np.argsort(-x,1); am=s[:,0]; age=9-am
T=m[:,None]-x; odd=(age%2==1); G9=np.maximum(x[:,9]-M[:,8],0)
np.set_printoptions(linewidth=200,suppress=True)
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
for i in range(16):
    y=h[:,i]
    r,c=fit(np.c_[m,m*odd,T,G9],y)
    w=c[2:12]
    # best 2-set partition by sorting coefs: contiguous split in sorted order, plus 3-set
    o=np.argsort(w); best=None
    for a in range(0,10):
        for b in range(a,10):
            S=[o[:a],o[a:b],o[b:]]
            fe=[T[:,g].sum(1) for g in S if len(g)]
            rr,cc=fit(np.c_[m,m*odd,np.stack(fe,1)] if len(fe)<=3 else None,y)
            if best is None or rr>best[0]: best=(rr,[sorted(g.tolist()) for g in S if len(g)],cc)
    print(i,'full %.3f'%r,'w',w.round(1),'M %.1f odd %.1f G9 %.1f'%(c[0],c[1],c[12]),'| 3set(5c) %.3f'%best[0],best[1],best[2].round(1))

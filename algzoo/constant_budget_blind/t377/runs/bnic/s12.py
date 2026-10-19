import numpy as np, itertools
x=np.load('x1.npy')[:50000]; h=np.load('h1.npy')[:50000]
M=np.maximum.accumulate(x,1); m=M[:,9]; s=np.argsort(-x,1); am=s[:,0]; age=9-am
T=m[:,None]-x; odd=(age%2==1); G9=np.maximum(x[:,9]-M[:,8],0)
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
base={'M,G9,odd':np.c_[m,G9,m*odd],'M,G9':np.c_[m,G9]}
for i in range(16):
    y=h[:,i]; res=[]
    for bn,B0 in base.items():
        nset=5-B0.shape[1]
        r,c=fit(np.c_[B0,T],y); w=c[B0.shape[1]:]; o=np.argsort(w)
        best=None
        for cuts in itertools.combinations(range(1,10),nset-1):
            S=np.split(o,cuts); fe=np.stack([T[:,g].sum(1) for g in S],1)
            rr,cc=fit(np.c_[B0,fe],y)
            if best is None or rr>best[0]: best=(rr,[sorted(g.tolist()) for g in S],cc)
        res.append((best[0],bn,best[1],best[2].round(2)))
    b=max(res,key=lambda t:t[0])
    print(i,' | '.join('%s %.3f'%(t[1],t[0]) for t in res),'BEST',b[1],b[2],b[3].tolist())

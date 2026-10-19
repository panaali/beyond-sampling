import numpy as np
from scipy.optimize import least_squares
x=np.load('x1.npy')[:40000];h=np.load('h1.npy')[:40000]
P=np.argmax(x[:,:9],1); s8=-np.sort(-x[:,:9],1); M8=s8[:,0];S8=s8[:,1];x9=x[:,9]
np.set_printoptions(precision=1,suppress=True,linewidth=150)
def kmeans1d(v,K):
    best=None
    vs=np.sort(v)
    # exhaustive over contiguous splits of sorted values
    import itertools
    n=len(v)
    for cuts in itertools.combinations(range(1,n),K-1):
        b=[0,*cuts,n];cost=sum(((vs[b[j]:b[j+1]]-vs[b[j]:b[j+1]].mean())**2).sum() for j in range(K))
        if best is None or cost<best[0]: best=(cost,b)
    b=best[1];thr=[vs[b[j]] for j in range(1,K)]
    return np.searchsorted(thr,v,side='right')
res={}
for i in range(16):
    hi=h[:,i]
    G=np.zeros(9)
    for p in range(9):
        m=P==p; X=np.c_[M8[m],S8[m],x9[m]]
        act=hi[m]>0
        if act.sum()<20: G[p]=0;continue
        c,*_=np.linalg.lstsq(X[act],hi[m][act],rcond=None);G[p]=c[0]
    cls=kmeans1d(G,3)
    def pred(th):
        g=th[:3][cls[P]]
        return np.maximum(0,g*M8+th[3]*S8+th[4]*x9)
    th0=np.r_[[G[cls==k].mean() for k in range(3)],0,0]
    r=least_squares(lambda th:pred(th)-hi,th0)
    r2=1-((pred(r.x)-hi)**2).mean()/hi.var()
    print(i,'G',G,'cls',cls,'th',np.round(r.x,2),'R2 %.3f'%r2)

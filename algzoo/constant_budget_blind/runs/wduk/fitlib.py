import numpy as np, itertools
from scipy.optimize import least_squares
def pf(x):
    o=np.argsort(-x[:,:9],1);s=-np.sort(-x[:,:9],1)
    return dict(P=o[:,0],Q=o[:,1],R=o[:,2],M=s[:,0],S=s[:,1],T=s[:,2],x9=x[:,9])
def clus(v,K):
    vs=np.sort(v);n=len(v);best=None
    for cuts in itertools.combinations(range(1,n),K-1):
        b=[0,*cuts,n];cost=sum(((vs[b[j]:b[j+1]]-vs[b[j]:b[j+1]].mean())**2).sum() for j in range(K))
        if best is None or cost<best[0]: best=(cost,b)
    b=best[1];thr=[(vs[b[j]-1]+vs[b[j]])/2 for j in range(1,K)]
    return np.searchsorted(thr,v)
def unconstrained(F,hi):
    E=np.eye(9);X=np.c_[E[F['P']]*F['M'][:,None],E[F['Q']]*F['S'][:,None],F['x9']]
    c,*_=np.linalg.lstsq(X,hi,rcond=None)
    r=least_squares(lambda th:np.maximum(0,X@th)-hi,c)
    return r.x[:9],r.x[9:18],r.x[18]
def model(name,F,th,cg,ch):
    P,Q,M,S,T,x9=F['P'],F['Q'],F['M'],F['S'],F['T'],F['x9']
    if name=='V1': z=th[:3][cg[P]]*M+th[3:5][ch[Q]]*S
    if name=='V2': g=th[:3][cg[P]]; z=g*M+(th[3]+th[4]*th[:3][cg[Q]])*S
    if name=='V3': z=th[:3][cg[P]]*M+th[3]*S+th[4]*x9
    if name=='V4': z=th[:2][cg[P]]*M+th[2:4][ch[Q]]*S+th[4]*x9
    if name=='V5': z=th[:3][cg[P]]*M+th[3]*S+th[4]*T
    return np.maximum(0,z)
def fit(name,F,hi,G,H):
    K=2 if name=='V4' else 3
    cg=clus(G,K); ch=clus(H,2)
    th0=np.r_[[G[cg==k].mean() if (cg==k).any() else 0 for k in range(K)]]
    if name=='V1': th0=np.r_[th0,[H[ch==k].mean() for k in range(2)]]
    elif name=='V4': th0=np.r_[th0,[H[ch==k].mean() for k in range(2)],0]
    else: th0=np.r_[th0,0,0]
    r=least_squares(lambda th:model(name,F,th,cg,ch)-hi,th0)
    return r.x,cg,ch
def r2(p,h): return 1-((p-h)**2).mean()/h.var()
def rankw(x,hi,R=4):
    o=np.argsort(-x[:,:9],1);s=-np.sort(-x[:,:9],1);E=np.eye(9)
    X=np.concatenate([E[o[:,r]]*s[:,r:r+1] for r in range(R)]+[x[:,9:10]],1)
    c,*_=np.linalg.lstsq(X,hi,rcond=None)
    r=least_squares(lambda th:np.maximum(0,X@th)-hi,c)
    return r.x[:-1].reshape(R,9)
def m9(x,F,th,cg,A,excl=True):
    mem=x[:,A].sum(1) if len(A) else 0*F['M']
    if excl and len(A): mem=mem-np.isin(F['P'],A)*F['M']
    return np.maximum(0,th[:3][cg[F['P']]]*F['M']+th[3]*F['S']+th[4]*mem)
def fit9(x,F,hi,G,A,excl=True):
    cg=clus(G,3)
    th0=np.r_[[G[cg==k].mean() for k in range(3)],0,0]
    r=least_squares(lambda th:m9(x,F,th,cg,A,excl)-hi,th0)
    return r.x,cg
def memf(x,F,A,tr,excl):
    v=x[:,A]
    if tr=='relu': v=np.maximum(v,0)
    if excl: v=v*(F['P'][:,None]!=np.array(A)[None,:])
    return v.sum(1)
def mB(x,F,th,cg,ch,A,tr,excl,mode):
    mem=memf(x,F,A,tr,excl)
    if mode=='g3': z=th[:3][cg[F['P']]]*F['M']+th[3]*F['S']+th[4]*mem
    else: z=th[:2][cg[F['P']]]*F['M']+th[2:4][ch[F['Q']]]*F['S']+th[4]*mem
    return np.maximum(0,z)
def fitB(x,F,hi,G,H,A,tr,excl,mode):
    if mode=='g3': cg=clus(G,3);ch=None;th0=np.r_[[G[cg==k].mean() for k in range(3)],0,0]
    else: cg=clus(G,2);ch=clus(H,2);th0=np.r_[[G[cg==k].mean() for k in range(2)],[H[ch==k].mean() for k in range(2)],0]
    r=least_squares(lambda th:mB(x,F,th,cg,ch,A,tr,excl,mode)-hi,th0)
    return r.x,cg,ch

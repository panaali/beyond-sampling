import numpy as np
from scipy.optimize import least_squares
R=lambda z: np.maximum(z,0)
def ns(x):
    d={'x':x,'R':R,'np':np}
    N=len(x); s=np.argsort(-x,1); srt=-np.sort(-x,1)
    d.update(am=s[:,0],a2=s[:,1],a3=s[:,2],T1=srt[:,0],T2=srt[:,1],T3=srt[:,2])
    p=x[:,:9]; s9=np.argsort(-p,1); sr9=-np.sort(-p,1)
    d.update(am9=s9[:,0],a29=s9[:,1],M9=sr9[:,0],S9=sr9[:,1],x9=x[:,9])
    def mo(k): o=np.delete(x,k,1); return o.max(1)
    def so(k): o=np.delete(x,k,1); return np.sort(o,1)[:,-2]
    def pm(k): return x[:,:k].max(1)       # max of x0..x_{k-1}
    def ps(k): return np.sort(x[:,:k],1)[:,-2]
    def qm(k): return x[:,k+1:].max(1)     # max of x_{k+1}..x9
    def isin(a,S): return np.isin(a,list(S)).astype(float)
    def rank(k): return (x>x[:,[k]]).sum(1)  # 0 = largest
    def P(L,j): return -np.sort(-x[:,:L],1)[:,j-1]
    def PA(L,j): return np.argsort(-x[:,:L],1)[:,j-1]
    def SM(L): return x[:,:L].sum(1)
    d.update(P=P,PA=PA,SM=SM)
    d.update(mo=mo,so=so,pm=pm,ps=ps,qm=qm,isin=isin,rank=rank,srt=srt)
    return d
def fit(expr,np_,h,d,p0=None,ht=None,dt=None):
    code=compile(expr,'e','eval')
    f=lambda p,d: eval(code,{'np':np,'R':R},{**d,'p':p})
    g=lambda p: f(p,d)
    p0=np.ones(np_)*0.5 if p0 is None else np.array(p0,float)
    r=least_squares(lambda p: g(p)-h,p0,method='trf',max_nfev=2000)
    p=r.x; r2=1-((g(p)-h)**2).mean()/h.var()
    r2t=None
    if ht is not None: r2t=1-((f(p,dt)-ht)**2).mean()/ht.var()
    return p,r2,r2t

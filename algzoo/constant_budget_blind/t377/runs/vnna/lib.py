import numpy as np
from scipy.optimize import least_squares
NEG=-10.0
def lib(x):
    F={}
    N=len(x); srt=-np.sort(-x,1)
    F['T1']=srt[:,0];F['T2']=srt[:,1];F['T3']=srt[:,2]
    for k in range(10):
        F['x%d'%k]=x[:,k]
        o=np.delete(x,k,1); so=-np.sort(-o,1)
        F['mo%d'%k]=so[:,0]; F['so%d'%k]=so[:,1]
        if k>0:
            pre=-np.sort(-x[:,:k],1); F['pm%d'%k]=pre[:,0]
            F['ps%d'%k]=pre[:,1] if k>1 else np.full(N,NEG)
        if k<9:
            post=-np.sort(-x[:,k+1:],1); F['qm%d'%k]=post[:,0]
            F['qs%d'%k]=post[:,1] if k<8 else np.full(N,NEG)
    return F
def relufit(h,X):
    X1=np.c_[X,np.ones(len(X))]
    m=h>0
    c0,*_=np.linalg.lstsq(X1[m],h[m],rcond=None)
    def res(c): return np.maximum(X1@c,0)-h
    r=least_squares(res,c0,method='lm' if len(c0)<len(h) else 'trf',max_nfev=200)
    c=r.x; p=np.maximum(X1@c,0)
    return c,1-((p-h)**2).mean()/h.var(),p
def greedy(h,F,names,steps=4,sub=6000):
    sel=[];out=[]
    hs=h[:sub]
    for s in range(steps):
        best=None
        for n in names:
            if n in sel: continue
            X=np.stack([F[q][:sub] for q in sel+[n]],1)
            try: c,r2,_=relufit(hs,X)
            except Exception: continue
            if best is None or r2>best[1]: best=(n,r2,c)
        sel.append(best[0]); out.append((list(sel),best[1],best[2]))
    return out

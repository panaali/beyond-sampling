import numpy as np, probe, sys, pickle
from lib import lib, NEG
from scipy.optimize import least_squares
def fastfit(h,X,it=6):
    m=h>0
    c,*_=np.linalg.lstsq(X[m],h[m],rcond=None)
    for _ in range(it):
        m=(h>0)|(X@c>0)
        c,*_=np.linalg.lstsq(X[m],h[m],rcond=None)
    p=np.maximum(X@c,0); return c,1-((p-h)**2).mean()/h.var()
def lmfit(h,X,c0):
    r=least_squares(lambda c: np.maximum(X@c,0)-h,c0,method='trf',max_nfev=300)
    return r.x
def build(x,h_ref=None,order=None):
    F=lib(x); F={k:v for k,v in F.items() if v.min()>NEG}
    s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]; srt=-np.sort(-x,1)
    if order is not None:
        for tag,(o,arr) in order.items():
            for j in range(1,10):
                S=o[:j]; g=np.isin(arr if tag[0]=='m' else arr,S)
                g=np.isin(am if tag[0]=='m' else a2,S)
                F['%s*[%s in %s]'%(tag[1:],'am' if tag[0]=='m' else 'a2',''.join(map(str,sorted(S))))]=(srt[:,0] if tag[1:]=='T1' else srt[:,1])*g
    return F,am,a2
if __name__=="__main__":
    rng=np.random.default_rng(int(sys.argv[1]))
    x=rng.standard_normal((5000,10)); H=probe.hidden(x)
    xt=np.random.default_rng(99).standard_normal((20000,10)); Ht=probe.hidden(xt)
    res={}
    for i in [int(a) for a in sys.argv[2:]]:
        h=H[:,i]
        s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
        mam=np.array([h[am==k].mean() for k in range(10)]); ma2=np.array([h[a2==k].mean() for k in range(10)])
        Zm=[k for k in range(10) if mam[k]<0.04*h.mean()]; Z2=[k for k in range(10) if ma2[k]<0.04*h.mean()]
        om=list(np.argsort(-mam)); o2=list(np.argsort(-ma2))
        order={'mT1':(om,None),'mT2':(om,None),'2T1':(o2,None),'2T2':(o2,None)}
        F,am,a2=build(x,order=order); names=list(F)
        gate=~np.isin(am,Zm)&~np.isin(a2,Z2)
        Ft,amt,a2t=build(xt,order=order); gt=~np.isin(amt,Zm)&~np.isin(a2t,Z2)
        sel=[]; hist=[]
        for step in range(5):
            best=None
            for n in names:
                if n in sel: continue
                X=np.stack([F[q] for q in sel+[n]],1)*gate[:,None]
                c,r2=fastfit(h,X)
                if best is None or r2>best[1]: best=(n,r2)
            sel.append(best[0])
            X=np.stack([F[q] for q in sel],1)*gate[:,None]
            c,_=fastfit(h,X); c=lmfit(h,X,c)
            Xt=np.stack([Ft[q] for q in sel],1)*gt[:,None]
            pt=np.maximum(Xt@c,0); r2t=1-((pt-Ht[:,i])**2).mean()/Ht[:,i].var()
            hist.append((list(sel),c,r2t))
        res[i]=dict(Zm=Zm,Z2=Z2,hist=hist)
        print(i,"Zam",Zm,"Za2",Z2,flush=True)
        for sl,c,r in hist: print("   %.3f"%r," + ".join("%.2f*%s"%(a,b) for a,b in zip(c,sl)),flush=True)
    pickle.dump(res,open('gate_%s.pkl'%sys.argv[2],'wb'))

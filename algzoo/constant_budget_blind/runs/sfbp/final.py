import numpy as np, sys, json, probe
from core import feat
def relufit(A,y,iters=20):
    m=y>1e-6
    for it in range(iters):
        co=np.linalg.lstsq(A[m],y[m],rcond=None)[0]; m=(y>1e-6)|(A@co>0)
    return co
def data(seed,N):
    x=np.random.default_rng(seed).standard_normal((N,10)); return x,probe.hidden(x)
xa,ha=data(11,30000); xb,hb=data(12,30000)
SPEC=json.load(open(sys.argv[1]))
out={}
for i,names in sorted(SPEC.items(),key=lambda t:int(t[0])):
    i=int(i)
    A=np.stack([feat(n,xa) for n in names],1); co=relufit(A,ha[:,i])
    B=np.stack([feat(n,xb) for n in names],1); pr=np.maximum(B@co,0)
    R=1-((pr-hb[:,i])**2).mean()/hb[:,i].var()
    out[i]=(names,[float(c) for c in co],float(R))
    print(i,'R2=%.3f'%R,list(zip(names,np.round(co,3))))
json.dump(out,open(sys.argv[2],'w'))

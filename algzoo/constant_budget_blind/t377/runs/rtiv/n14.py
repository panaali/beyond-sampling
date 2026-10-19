import numpy as np
from scipy.optimize import least_squares
d=np.load('d1.npz');x=d['x'][:50000];H=d['H'][:50000];xv=d['x'][50000:];Hv=d['H'][50000:]
i=14
def F(x):
    o=np.argsort(-x,1);am=o[:,0];s=-np.sort(-x,1);pm=np.maximum.accumulate(x,1)
    g=(am%2==1).astype(float)
    return dict(am=am,M=s[:,0],S=s[:,1],g=g,pm3=pm[:,3],pm8=pm[:,8],x=x)
def ev(X,Xv,name):
    c0,*_=np.linalg.lstsq(X,H[:,i],rcond=None)
    c=least_squares(lambda c: np.maximum(X@c,0)-H[:,i],c0).x
    p=np.maximum(Xv@c,0);print(name,"%.3f"%(1-((Hv[:,i]-p)**2).mean()/Hv[:,i].var()),np.round(c,2))
f=F(x);fv=F(xv)
mk=lambda f: np.c_[f['g']*f['M'],f['g']*f['S'],(1-f['g'])*f['M'],(1-f['g'])*f['S'],f['x'][:,4],f['x'][:,9]]
ev(mk(f),mk(fv),"par MS x4 x9")
mk=lambda f: np.c_[f['M'],f['S'],f['pm3'],f['x'][:,4],f['x'][:,9]]
ev(mk(f),mk(fv),"M S pm3 x4 x9")
mk=lambda f: np.c_[f['x'],f['M'],f['S'],f['g']*f['M']]
ev(mk(f),mk(fv),"relu(all x, M,S,gM)")
# mean by am and by a2
o=np.argsort(-x,1)
print("mean by am",[round(H[o[:,0]==k,i].mean(),1) for k in range(10)])
print("mean by a2",[round(H[o[:,1]==k,i].mean(),1) for k in range(10)])
print("frac zero by am",[round((H[o[:,0]==k,i]==0).mean(),2) for k in range(10)])

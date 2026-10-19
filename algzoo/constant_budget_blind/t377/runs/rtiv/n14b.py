import numpy as np
from scipy.optimize import least_squares
d=np.load('d1.npz');x=d['x'][:30000];H=d['H'][:30000];xv=d['x'][60000:90000];Hv=d['H'][60000:90000]
def lib(x):
    o=np.argsort(-x,1);am=o[:,0];s=-np.sort(-x,1);pm=np.maximum.accumulate(x,1);M=s[:,0];S=s[:,1]
    od=(am%2==1)*1.;late=(am>4)*1.
    D={'M':M,'S':S,'odM':od*M,'odS':od*S,'lateM':late*M,'lateS':late*S,'pm3':pm[:,3],'pm6':pm[:,6],'pm8':pm[:,8],'S8':-np.sort(-x[:,:9],1)[:,1]}
    for t in range(10): D[f'x{t}']=x[:,t]
    return D
D=lib(x);Dv=lib(xv);names=list(D)
def run(i,start,k=5):
    y=H[:,i];yv=Hv[:,i];sel=list(start)
    while len(sel)<k:
        best=None
        for n in names:
            if n in sel: continue
            X=np.stack([D[m] for m in sel+[n]],1);c0,*_=np.linalg.lstsq(X,y,rcond=None)
            c=least_squares(lambda c: np.maximum(X@c,0)-y,c0,max_nfev=50).x
            Xv=np.stack([Dv[m] for m in sel+[n]],1);r=1-((yv-np.maximum(Xv@c,0))**2).mean()/yv.var()
            if best is None or r>best[0]: best=(r,n,c)
        sel.append(best[1]);print(i,sel,"%.3f"%best[0],np.round(best[2],2))
import sys
for a in sys.argv[1:]:
    i,t=a.split(":");run(int(i),[t])

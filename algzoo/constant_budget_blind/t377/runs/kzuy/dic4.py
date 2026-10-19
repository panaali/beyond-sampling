import numpy as np
from dic3 import dic3
def dic4(x):
    D=dic3(x)
    s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1); a2=np.argsort(-x,1)[:,1]
    E=(am%2==0)&(am<8); O=(am%2==1)&(am<8)
    D['PE*s0']=E*s[:,0]; D['PO*s0']=O*s[:,0]; D['L*s0']=(am<8)*s[:,0]
    D['PE*s1']=E*s[:,1]; D['PO*s1']=O*s[:,1]
    for t in range(10):
        D[f'xn{t}']=x[:,t]*(am!=t)
        D[f'xn2{t}']=x[:,t]*(am!=t)*(a2!=t)
    return D
def greedyR(D,y,k=5,it=8):
    keys=list(D); X=np.stack([D[q] for q in keys],1); X=X/(X.std(0)+1e-9)
    ch=[];out=[]
    for _ in range(k):
        best=None
        for j in range(len(keys)):
            if j in ch: continue
            A=X[:,ch+[j]]; b=np.linalg.lstsq(A,y,rcond=None)[0]
            for _ in range(it):
                m=(A@b>0)|(y>0); b=np.linalg.lstsq(A[m],y[m],rcond=None)[0]
            e=((y-np.maximum(A@b,0))**2).mean()
            if best is None or e<best[0]: best=(e,j)
        ch.append(best[1]); out.append((keys[best[1]],1-best[0]/y.var()))
    return out

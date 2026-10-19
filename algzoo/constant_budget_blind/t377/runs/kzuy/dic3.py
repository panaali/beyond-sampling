import numpy as np
from dic2 import dic2
r=lambda z:np.maximum(z,0)
def dic3(x):
    D=dic2(x)
    for k in [k for k in D if k.startswith('am') and '==' in k or k.startswith('a2==') or k.startswith('am>=') or k.startswith('a2>=') or k in('am','a2')]: D.pop(k)
    s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1); M=np.maximum.accumulate(x,1)
    for t in range(10):
        D[f'am6*x{t}']=(am==6)*x[:,t]
    D['am6*s1']=(am==6)*s[:,1]
    for k in [7,8]:
        a=np.argmax(x[:,:k+1],1); D[f'po{k}*M{k}']=(a%2==1)*M[:,k]
    return D
def greedy0(D,y,k=5,keys=None):
    keys=keys or list(D); X=np.stack([D[q] for q in keys],1); sc=X.std(0)+1e-9; X=X/sc
    ch=[]; out=[]
    for _ in range(k):
        best=None
        for j in range(len(keys)):
            if j in ch: continue
            A=X[:,ch+[j]]; b=np.linalg.lstsq(A,y,rcond=None)[0]; e=((y-A@b)**2).mean()
            if best is None or e<best[0]: best=(e,j)
        ch.append(best[1]); out.append((keys[best[1]],1-best[0]/y.var()))
    return out
def fitrelu(X,y,it=30):
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    for _ in range(it):
        p=X@b; m=(p>0)|(y>0)
        b=np.linalg.lstsq(X[m],y[m],rcond=None)[0]
    return b

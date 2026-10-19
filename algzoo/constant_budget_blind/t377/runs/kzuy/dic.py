import numpy as np
r=lambda z:np.maximum(z,0)
def dic(x):
    N=x.shape[0]; D={}
    M=np.maximum.accumulate(x,1)
    S=np.full_like(x,-np.inf)
    for t in range(1,10):
        S[:,t]=np.sort(x[:,:t+1],1)[:,-2]
    s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1); a2=np.argsort(-x,1)[:,1]
    for t in range(10):
        D[f'x{t}']=x[:,t]; D[f'M{t}']=M[:,t]; D[f'M9-x{t}']=M[:,9]-x[:,t]
        D[f'M{t}-x{t}']=M[:,t]-x[:,t]
        D[f'am=={t}']=(am==t)*1.; D[f'a2=={t}']=(a2==t)*1.
        D[f'am>={t}']=(am>=t)*1.
        D[f'a2>={t}']=(a2>=t)*1.
        D[f'r(x{t}-m2)']=r(x[:,t]-s[:,1]) ; D[f'r(m2-x{t})']=r(s[:,1]-x[:,t])
        if t>0:
            D[f'r(x{t}-M{t-1})']=r(x[:,t]-M[:,t-1]); D[f'S{t}']=S[:,t]; D[f'M{t}-S{t}']=M[:,t]-S[:,t]
            D[f'r(x{t}-S{t-1})']=r(x[:,t]-S[:,t-1]) if t>1 else r(x[:,t]-x[:,0])
            D[f'r(M{t-1}-x{t})']=r(M[:,t-1]-x[:,t])
    for k in range(4): D[f's{k}']=s[:,k]
    D['g01']=s[:,0]-s[:,1]; D['g12']=s[:,1]-s[:,2]
    D['am']=am*1.; D['a2']=a2*1.
    return D
def greedy(D,y,k=4):
    keys=list(D); X=np.stack([D[q] for q in keys],1); X=(X-X.mean(0))/(X.std(0)+1e-9)
    ch=[]; yc=y-y.mean(); out=[]
    for _ in range(k):
        best=None
        for j in range(len(keys)):
            if j in ch: continue
            A=X[:,ch+[j]]; b=np.linalg.lstsq(A,yc,rcond=None)[0]; e=((yc-A@b)**2).mean()
            if best is None or e<best[0]: best=(e,j)
        ch.append(best[1]); out.append((keys[best[1]],1-best[0]/yc.var()))
    return out

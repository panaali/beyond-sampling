import numpy as np
def dictionary(x):
    D={}
    N=len(x); s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
    M=np.maximum.accumulate(x,1)
    for k in range(10): D[f'x{k}']=x[:,k]
    for k in range(10): D[f'M{k}']=M[:,k]
    for k in range(1,10):
        D[f'relu(x{k}-M{k-1})']=np.maximum(x[:,k]-M[:,k-1],0)
        D[f'relu(M{k-1}-x{k})']=np.maximum(M[:,k-1]-x[:,k],0)
    D['s0']=s[:,0];D['s1']=s[:,1];D['s2']=s[:,2]
    for k in range(10):
        D[f's0*[am={k}]']=s[:,0]*(am==k); D[f's1*[a2={k}]']=s[:,1]*(a2==k)
        D[f's0*[a2={k}]']=s[:,0]*(a2==k); D[f's1*[am={k}]']=s[:,1]*(am==k)
    pe=(am%2==0); D['s0*[am even]']=s[:,0]*pe; D['s0*[am odd]']=s[:,0]*(~pe)
    D['s1*[a2 even]']=s[:,1]*(a2%2==0)
    return D
def greedy(D,y,k=5):
    names=list(D); A=np.stack([D[n] for n in names],1); A=A/ (A.std(0)+1e-9)
    sel=[]; 
    for _ in range(k):
        best=None
        for j in range(A.shape[1]):
            if j in sel: continue
            B=A[:,sel+[j]]; c,*_=np.linalg.lstsq(B,y,rcond=None); r=((B@c-y)**2).mean()
            if best is None or r<best[0]: best=(r,j)
        sel.append(best[1])
    B=np.stack([D[names[j]] for j in sel],1); c,*_=np.linalg.lstsq(B,y,rcond=None)
    r2=1-((B@c-y)**2).mean()/y.var()
    return [names[j] for j in sel],c,r2
def dict2(x):
    D=dictionary(x); N=len(x)
    s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
    # running second max
    S=np.full((N,10),-np.inf)
    for k in range(1,10):
        S[:,k]=np.sort(x[:,:k+1],1)[:,-2]
    for k in range(1,10): D[f'S{k}']=S[:,k]
    D['gap']=s[:,0]-s[:,1]
    for k in range(10): D[f'relu(x{k}-maxothers)']=np.maximum(x[:,k]-np.max(np.delete(x,k,1),1),0)
    for k in range(10): D[f'relu(x{k}-maxprev..)']=x[:,k]*(am==k)
    D['am']=am*s[:,0]/5; 
    D['gap*[am odd]']=(s[:,0]-s[:,1])*(am%2==1); D['gap*[am even]']=(s[:,0]-s[:,1])*(am%2==0)
    D['s1*[a2 odd]']=s[:,1]*(a2%2==1)
    for k in range(10): D[f'gap*[am={k}]']=(s[:,0]-s[:,1])*(am==k)
    return D

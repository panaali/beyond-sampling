import numpy as np
def plib(x, n=9):
    # features of prefix x[:, :n]
    x=x[:,:n]; N=len(x); F={}; r=lambda a: np.maximum(a,0)
    srt=-np.sort(-x,1); s=np.argsort(-x,1)
    for j in range(n): F[f'x{j}']=x[:,j]; F[f'r{j}']=r(x[:,j])
    for k in range(4): F[f'Q{k+1}']=srt[:,k]
    P=np.maximum.accumulate(x,1)
    for t in range(n-1): F[f'P{t}']=P[:,t]
    for j in range(1,n):
        F[f'rnew{j}']=r(x[:,j]-P[:,j-1]); F[f'rbelow{j}']=r(P[:,j-1]-x[:,j])
    F['Q1-Q2']=srt[:,0]-srt[:,1]; F['Q2-Q3']=srt[:,1]-srt[:,2]
    for j in range(n):
        F[f'1max{j}*Q1']=(s[:,0]==j)*srt[:,0]
        F[f'1sec{j}*Q2']=(s[:,1]==j)*srt[:,1]
        F[f'1max{j}*Q1-Q2']=(s[:,0]==j)*(srt[:,0]-srt[:,1])
        F[f'1sec{j}*Q2-Q3']=(s[:,1]==j)*(srt[:,1]-srt[:,2])
        F[f'x{j}-Q1']=x[:,j]-srt[:,0]
        F[f'x{j}-Q2']=x[:,j]-srt[:,1]
        F[f'r(x{j}-Q2)']=r(x[:,j]-srt[:,1])
        F[f'r(x{j}-Q3)']=r(x[:,j]-srt[:,2])
    return F
def greedy(F,y,steps=4,mask=None):
    names=list(F); A=np.stack([F[n] for n in names],1)
    if mask is not None: A=A[mask]; y=y[mask]
    chosen=[]; out=[]
    for st in range(steps):
        best=None
        for k in range(A.shape[1]):
            if k in chosen: continue
            X=np.c_[A[:,chosen+[k]],np.ones(len(y))]
            c=np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-((X@c-y)**2).mean()/y.var()
            if best is None or r2>best[0]: best=(r2,k,c)
        chosen.append(best[1]); out.append(f'{names[best[1]]}:{best[0]:.3f}')
    return ' '.join(out), best[2]

import numpy as np
r=lambda a: np.maximum(a,0)
def flib(x):
    N,n=x.shape; F={}
    srt=-np.sort(-x,1); s=np.argsort(-x,1)
    rank=np.argsort(s,1)  # rank of each position (0=largest)
    for k in range(4): F[f'Q{k+1}']=srt[:,k]
    F['Q1-Q2']=srt[:,0]-srt[:,1]; F['Q2-Q3']=srt[:,1]-srt[:,2]
    for j in range(n):
        xj=x[:,j]
        oth=np.delete(x,j,1); os_=-np.sort(-oth,1)
        F[f'x{j}']=xj; F[f'r{j}']=r(xj)
        F[f'x{j}-O1']=xj-os_[:,0]; F[f'r(x{j}-O1)']=r(xj-os_[:,0])
        F[f'x{j}-O2']=xj-os_[:,1]; F[f'r(x{j}-O2)']=r(xj-os_[:,1])
        F[f'r(x{j}-O3)']=r(xj-os_[:,2])
        F[f'r(O1-x{j})']=r(os_[:,0]-xj); F[f'r(O2-x{j})']=r(os_[:,1]-xj)
        if j<n-1:
            L=x[:,j+1:].max(1); F[f'r(x{j}-L)']=r(xj-L)
        if j>0:
            E=x[:,:j].max(1); F[f'r(x{j}-E)']=r(xj-E); F[f'r(E-x{j})']=r(E-xj)
        F[f'1max{j}*Q1']=(s[:,0]==j)*srt[:,0]; F[f'1sec{j}*Q2']=(s[:,1]==j)*srt[:,1]
        F[f'1max{j}*(Q1-Q2)']=(s[:,0]==j)*(srt[:,0]-srt[:,1])
        F[f'1sec{j}*(Q2-Q3)']=(s[:,1]==j)*(srt[:,1]-srt[:,2])
        F[f'1top2_{j}*Q2']=(rank[:,j]<2)*srt[:,1]
    return F
def greedy(F,y,steps=4,mask=None,names=None):
    names=names or list(F); A=np.stack([F[n] for n in names],1)
    yy=y if mask is None else y[mask]; AA=A if mask is None else A[mask]
    chosen=[]; out=[]
    for st in range(steps):
        best=None
        for k in range(A.shape[1]):
            if k in chosen: continue
            X=np.c_[AA[:,chosen+[k]],np.ones(len(yy))]
            c=np.linalg.lstsq(X,yy,rcond=None)[0]; r2=1-((X@c-yy)**2).mean()/yy.var()
            if best is None or r2>best[0]: best=(r2,k,c)
        chosen.append(best[1])
        # full relu R2
        X=np.c_[A[:,chosen],np.ones(len(y))]; pr=r(X@best[2]); R=1-((pr-y)**2).mean()/y.var()
        out.append(f'{names[best[1]]}:{best[0]:.2f}/{R:.2f}')
    return ' '.join(out), best[2]

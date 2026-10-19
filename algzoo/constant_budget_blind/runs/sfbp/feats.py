import numpy as np
def lib(x):
    N=len(x); F={}
    r=lambda a: np.maximum(a,0)
    srt=-np.sort(-x,1); s=np.argsort(-x,1)
    for j in range(10): F[f'x{j}']=x[:,j]
    for k in range(4): F[f'M{k+1}']=srt[:,k]
    P=np.maximum.accumulate(x,1)
    for t in range(10): F[f'P{t}']=P[:,t]   # max x0..xt
    S=np.full((N,),-np.inf); m=x[:,0].copy(); S2=[]
    sec=np.full(N,-np.inf)
    for t in range(1,10):
        sec=np.where(x[:,t]>m, m, np.maximum(sec,x[:,t])); m=np.maximum(m,x[:,t])
        F[f'S{t}']=sec.copy()  # second max of x0..xt
    for j in range(1,10):
        F[f'rnew{j}']=r(x[:,j]-P[:,j-1]); F[f'rbelow{j}']=r(P[:,j-1]-x[:,j])
    F['r_x9_S8']=r(x[:,9]-F['S8'])
    F['M1-M2']=srt[:,0]-srt[:,1]; F['M2-M3']=srt[:,1]-srt[:,2]
    for j in range(10):
        F[f'1max{j}*M1']=(s[:,0]==j)*srt[:,0]
        F[f'1sec{j}*M2']=(s[:,1]==j)*srt[:,1]
    for k in ['x','M']: pass
    for j in range(10):
        F[f'r{j}']=r(x[:,j])
    return F

import numpy as np
from lib2 import fitrelu, R
def library(x):
    y=x[:,:9]; s=np.sort(y,1)[:,::-1]; a=np.argsort(-y,1); A,B,C=s[:,0],s[:,1],s[:,2]; pA,pB=a[:,0],a[:,1]; x9=x[:,9]
    F={'A':A,'B':B,'C':C,'x9':x9,'x9-A+':R(x9-A),'x9-B+':R(x9-B),'A-x9+':R(A-x9),'B-x9+':R(B-x9),'C-x9+':R(C-x9),'x9-C+':R(x9-C)}
    for j in range(9):
        F['x%d'%j]=x[:,j]
        F['A@%d'%j]=A*(pA==j); F['B@%d'%j]=B*(pB==j)
        F['A@<=%d'%j]=A*(pA<=j); F['B@<=%d'%j]=B*(pB<=j)
        F['xj-B+%d'%j]=R(x[:,j]-B); F['B-xj+%d'%j]=R(B-x[:,j]); F['C-xj+%d'%j]=R(C-x[:,j])
        F['x9@A%d'%j]=x9*(pA==j); F['x9@B%d'%j]=x9*(pB==j)
    m=np.maximum.accumulate(x,1)
    for t in range(1,9): F['m%d'%t]=m[:,t]; F['m2_%d'%t]=np.sort(x[:,:t+1],1)[:,-2]
    return F
def greedy(F,y,k=4):
    K=list(F); sel=[]; out=[]
    for _ in range(k):
        best=None
        for q in K:
            if q in sel: continue
            X=np.array([F[z] for z in sel+[q]]).T
            r2,c=fitrelu(X,y,3)
            if best is None or r2>best[0]: best=(r2,q,c)
        sel.append(best[1]); out.append((best[1],round(best[0],3)))
    return out,best[2]

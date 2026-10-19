import numpy as np
def feats(x):
    F={}; W=x[:,:9]
    s=-np.sort(-W,1); o=np.argsort(-W,1); rank=np.argsort(o,1)
    M,S,T=s[:,0],s[:,1],s[:,2]; F['M9']=M;F['S9']=S;F['T9']=T; F['x9']=x[:,9]
    F['x9-M9']=x[:,9]-M; F['x9-S9']=x[:,9]-S; F['x9-T9']=x[:,9]-T
    F['r(x9-M9)']=np.maximum(x[:,9]-M,0); F['r(x9-S9)']=np.maximum(x[:,9]-S,0); F['r(x9-T9)']=np.maximum(x[:,9]-T,0)
    F['r(S9-x9)']=np.maximum(S-x[:,9],0); F['r(M9-x9)']=np.maximum(M-x[:,9],0); F['r(T9-x9)']=np.maximum(T-x[:,9],0)
    for j in range(9):
        oth=np.delete(W,j,1); so=-np.sort(-oth,1); xj=W[:,j]
        F['x%d'%j]=xj
        F['x%d-mo'%j]=xj-so[:,0]; F['x%d-so'%j]=xj-so[:,1]; F['x%d-to'%j]=xj-so[:,2]
        F['r(x%d-mo)'%j]=np.maximum(xj-so[:,0],0); F['r(x%d-so)'%j]=np.maximum(xj-so[:,1],0); F['r(x%d-to)'%j]=np.maximum(xj-so[:,2],0)
        F['r(so-x%d)'%j]=np.maximum(so[:,1]-xj,0); F['r(mo-x%d)'%j]=np.maximum(so[:,0]-xj,0)
        F['am%d'%j]=(rank[:,j]==0)*1.; F['a2%d'%j]=(rank[:,j]==1)*1.; F['top2_%d'%j]=(rank[:,j]<=1)*1.
        if j>0: F['r(x%d-earl)'%j]=np.maximum(xj-W[:,:j].max(1),0)
        if j<8: F['r(x%d-lat)'%j]=np.maximum(xj-W[:,j+1:].max(1),0)
    return F
def greedy(X,y,names,k=5,pre=()):
    A=np.c_[X,np.ones(len(y))]; G=A.T@A; b=A.T@y; n=A.shape[1]-1; yy=(y**2).sum(); N=len(y)
    cols=list(pre); hist=[]
    for step in range(k):
        best=None
        for c in range(n):
            if c in cols: continue
            idx=cols+[c,n]; 
            try: w=np.linalg.solve(G[np.ix_(idx,idx)]+1e-9*np.eye(len(idx)),b[idx])
            except: continue
            sse=yy-w@b[idx]; r=1-sse/N/y.var()
            if best is None or r>best[0]: best=(r,c)
        cols.append(best[1]); hist.append((names[best[1]],round(best[0],3)))
    return cols,hist
def relufit(X,cols,y,it=10):
    A=np.c_[X[:,cols],np.ones(len(y))]; m=np.ones(len(y),bool)
    for _ in range(it):
        c,*_=np.linalg.lstsq(A[m],y[m],rcond=None); m=(y>1e-6)|(A@c>0)
    p=np.maximum(A@c,0); return 1-((p-y)**2).mean()/y.var(),c

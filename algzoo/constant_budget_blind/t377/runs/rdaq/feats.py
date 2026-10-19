import numpy as np
def F(x):
    N=x.shape[0]
    P=np.maximum.accumulate(x,1)
    s=-np.sort(-x,1)
    srt=np.argsort(-x,1); am=srt[:,0]; a2=srt[:,1]
    rec=np.stack([np.maximum(x[:,j]-P[:,j-1],0) for j in range(1,10)],1)
    d=dict(x=x,P=P,s=s[:,:4],am=np.eye(10)[am],a2=np.eye(10)[a2],rec=rec)
    return d
def r2(X,y):
    X=np.hstack([X,np.ones((len(y),1))])
    c,*_=np.linalg.lstsq(X,y,rcond=None)
    return 1-((y-X@c)**2).mean()/y.var(),c
def Q(x):
    return np.maximum.accumulate(x[:,::-1],1)[:,::-1]
def fit_relu(X,y,it=30):
    Xb=np.hstack([X,np.ones((len(y),1))])
    c,*_=np.linalg.lstsq(Xb,y,rcond=None)
    for _ in range(it):
        z=Xb@c
        m=(y>1e-6)|(z>0)
        c,*_=np.linalg.lstsq(Xb[m],y[m],rcond=None)
    p=np.maximum(Xb@c,0)
    return 1-((y-p)**2).mean()/y.var(),c
def prefix_top(x,L,k):
    return -np.sort(-x[:,:L],1)[:,k]
def big(x):
    cols=[x[:,j] for j in range(10)]; names=['x%d'%j for j in range(10)]
    for L in range(2,11):
        cols.append(prefix_top(x,L,0)); names.append('M%d'%L)
        cols.append(prefix_top(x,L,1)); names.append('S%d'%L)
    return np.column_stack(cols),names
def catalog(x):
    C={}
    N=x.shape[0]
    for j in range(10): C['x%d'%j]=x[:,j]
    for L in range(2,11):
        C['M%d'%L]=prefix_top(x,L,0); C['S%d'%L]=prefix_top(x,L,1)
    for L in range(3,11): C['T%d'%L]=prefix_top(x,L,2)
    for j in range(10):
        others=np.delete(x,j,1)
        so=-np.sort(-others,1)
        C['Mo%d'%j]=so[:,0]; C['So%d'%j]=so[:,1]
        if j>=1: C['Mb%d'%j]=x[:,:j].max(1)
        if j<=8: C['Ma%d'%j]=x[:,j+1:].max(1)
        if j<=7: C['Ma8_%d'%j]=x[:,j+1:9].max(1)
    C['sum']=x.sum(1); C['sum9']=x[:,:9].sum(1)
    return C
def greedy(C,y,k=4,it=10):
    names=list(C); chosen=[]; best=None
    for step in range(k):
        cand=[]
        for n in names:
            if n in chosen: continue
            X=np.column_stack([C[m] for m in chosen+[n]])
            r,c=fit_relu(X,y,it); cand.append((r,n,c))
        cand.sort(key=lambda t:-t[0])
        chosen.append(cand[0][1]); best=cand[0]
        yield list(chosen),best[0],best[2]
def fit_gated(X,y,gate,it=15):
    # y ~ gate*ReLU(Xc)
    r,c=fit_relu(X[gate],y[gate],it)
    Xb=np.hstack([X,np.ones((len(y),1))])
    p=np.maximum(Xb@c,0)*gate
    return 1-((y-p)**2).mean()/y.var(),c

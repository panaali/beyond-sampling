import numpy as np
R=lambda z: np.maximum(z,0)
def topk_others(x,t,k):
    o=np.delete(x,t,1); return -np.sort(-o,1)[:,:k]
def build(x):
    N,L=x.shape; F={}; x9={}
    def add(n,v,u9): F[n]=v.astype(float); x9[n]=u9
    s=-np.sort(-x,1); o=np.argsort(-x,1)
    pre=x[:,:9]; ps=-np.sort(-pre,1); po=np.argsort(-pre,1)
    P=np.maximum.accumulate(x,1)
    for k in range(4): add('s%d'%(k+1),s[:,k],True)
    for k in range(3): add('ps%d'%(k+1),ps[:,k],False)
    for t in range(10):
        u=(t==9)
        add('x%d'%t,x[:,t],u)
        add('[am=%d]'%t,o[:,0]==t,True); add('[a2=%d]'%t,o[:,1]==t,True)
        ot=topk_others(x,t,3)
        add('R(x%d-o1)'%t,R(x[:,t]-ot[:,0]),True); add('R(x%d-o2)'%t,R(x[:,t]-ot[:,1]),True); add('R(x%d-o3)'%t,R(x[:,t]-ot[:,2]),True)
        add('R(o1-x%d)'%t,R(ot[:,0]-x[:,t]),True); add('R(o2-x%d)'%t,R(ot[:,1]-x[:,t]),True)
        if t>=1:
            add('P%d'%(t-1),P[:,t-1],False)
            add('R(x%d-P%d)'%(t,t-1),R(x[:,t]-P[:,t-1]),u); add('R(P%d-x%d)'%(t-1,t),R(P[:,t-1]-x[:,t]),u)
        if t<9:
            add('[pam=%d]'%t,po[:,0]==t,False); add('[pa2=%d]'%t,po[:,1]==t,False)
            pt=topk_others(pre,t,2)
            add('R(x%d-po1)'%t,R(x[:,t]-pt[:,0]),False); add('R(x%d-po2)'%t,R(x[:,t]-pt[:,1]),False)
            add('R(po1-x%d)'%t,R(pt[:,0]-x[:,t]),False)
    add('R(x9-ps2)',R(x[:,9]-ps[:,1]),True); add('R(ps2-x9)',R(ps[:,1]-x[:,9]),True)
    return F,x9
def greedy(y,F,names,kmax=4):
    sel=[]; Y=y-y.mean()
    hist=[]
    for k in range(kmax):
        best=None
        for n in names:
            if n in sel: continue
            A=np.c_[np.stack([F[m] for m in sel+[n]],1),np.ones(len(y))]
            c,*_=np.linalg.lstsq(A,y,rcond=None); r=y-A@c; r2=1-r.var()/y.var()
            if best is None or r2>best[0]: best=(r2,n,c)
        sel.append(best[1]); hist.append((best[0],list(sel),best[2]))
    return hist
def build2(x):
    F,u9=build(x)
    for t in range(2,10):
        q=-np.sort(-x[:,:t],1)[:,:2]
        F['Q2_%d'%t]=q[:,1]; u9['Q2_%d'%t]=False
        F['R(Q2_%d-x%d)'%(t,t)]=R(q[:,1]-x[:,t]); u9['R(Q2_%d-x%d)'%(t,t)]=(t==9)
        F['R(x%d-Q2_%d)'%(t,t)]=R(x[:,t]-q[:,1]); u9['R(x%d-Q2_%d)'%(t,t)]=(t==9)
    return F,u9
def relufit(y,A,iters=6):
    m=y>1e-6
    c,*_=np.linalg.lstsq(A[m],y[m],rcond=None)
    for _ in range(iters):
        m=(y>1e-6)|(A@c>0)
        c,*_=np.linalg.lstsq(A[m],y[m],rcond=None)
    return c
def greedy_act(y,F,names,kmax=4):
    m=y>1e-6; sel=[]; hist=[]
    for k in range(kmax):
        best=None
        for n in names:
            if n in sel: continue
            A=np.c_[np.stack([F[s][m] for s in sel+[n]],1),np.ones(m.sum())]
            c,*_=np.linalg.lstsq(A,y[m],rcond=None); r=y[m]-A@c
            # penalize predicted positive on inactive
            Af=np.c_[np.stack([F[s] for s in sel+[n]],1),np.ones(len(y))]
            p=np.maximum(Af@c,0); r2=1-((y-p)**2).mean()/y.var()
            if best is None or r2>best[0]: best=(r2,n)
        sel.append(best[1])
        A=np.c_[np.stack([F[s] for s in sel],1),np.ones(len(y))]
        c=relufit(y,A); p=np.maximum(A@c,0)
        hist.append((1-((y-p)**2).mean()/y.var(),list(sel),c))
    return hist
def build3(x):
    F,u9=build2(x)
    for t in range(0,8):
        for k in range(t+1,9):
            others=np.delete(x[:,:k+1],t,1).max(1)
            n='R(x%d-m[0..%d]\\%d)'%(t,k,t); F[n]=R(x[:,t]-others); u9[n]=False
            if t>0:
                o2=x[:,t+1:k+1].max(1); n='R(x%d-m[%d..%d])'%(t,t+1,k); F[n]=R(x[:,t]-o2); u9[n]=False
    for k in range(1,9):
        n='m[1..%d]'%k; F[n]=x[:,1:k+1].max(1); u9[n]=False
    return F,u9

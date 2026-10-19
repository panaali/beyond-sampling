import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
o=np.argsort(-x,1); s=-np.sort(-x,1)
am=o[:,0]; a2=o[:,1]; m1=s[:,0]; m2=s[:,1]; m3=s[:,2]
rm=np.maximum.accumulate(x,1)   # rm[:,j]=max(x0..xj)
pm=np.concatenate([np.full((len(x),1),-np.inf),rm[:,:-1]],1) # max before j
relu=lambda z:np.maximum(z,0)
def r2(y,p): return 1-((y-p)**2).mean()/y.var()
def fit(y,cols,icpt=False):
    A=np.stack(cols,1)
    if icpt: A=np.concatenate([A,np.ones((len(y),1))],1)
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    return c, r2(y,A@c)
def relufit(y,A,it=8):
    act=y>0
    for _ in range(it):
        c,*_=np.linalg.lstsq(A[act],y[act],rcond=None)
        p=A@c; act=(p>0)|(y>0)
        # samples with y==0 and p>0 should be pushed: include them with target 0 (already)
    return c, r2(y,relu(A@c))
names=['x%d'%j for j in range(10)]+['M%d'%j for j in range(1,9)]+['m1','m2','m3']
Abank=np.concatenate([x,rm[:,1:9],m1[:,None],m2[:,None],m3[:,None]],1)
N=len(x)
S=np.zeros((N,10)); T=np.zeros((N,10))
for k in range(10):
    ss=-np.sort(-x[:,:k+1],1)
    S[:,k]=ss[:,1] if k>=1 else -9
    T[:,k]=ss[:,2] if k>=2 else -9
R=np.zeros((N,10))
for k in range(1,10): R[:,k]=relu(x[:,k]-rm[:,k-1])
names2=['x%d'%j for j in range(10)]+['M%d'%j for j in range(1,10)]+['S%d'%j for j in range(1,10)]+['T%d'%j for j in range(2,10)]+['R%d'%j for j in range(1,10)]
B2=np.concatenate([x,rm[:,1:],S[:,1:],T[:,2:],R[:,1:]],1)
def greedy(y,A,nm,k=5,sub=30000):
    yy=y[:sub]; AA=A[:sub]; sel=[]
    for step in range(k):
        best=None
        for j in range(A.shape[1]):
            if j in sel: continue
            c,r=relufit(yy,AA[:,sel+[j]],it=5)
            if best is None or r>best[1]: best=(j,r)
        sel.append(best[0])
    c,r=relufit(y,A[:,sel])
    return [(nm[j],round(float(cc),3)) for j,cc in zip(sel,c)], r
def pbank(p):
    xp=x[:,p]
    bef=x[:,:p].max(1) if p>0 else np.full(N,-3.)
    aft=x[:,p+1:].max(1) if p<9 else np.full(N,-3.)
    sb=x[:,:p].sum(1); sa=x[:,p+1:].sum(1)
    xo=np.delete(x,p,1); so=-np.sort(-xo,1)
    g=(am==p).astype(float); g2=(a2==p).astype(float)
    Rp=relu(xp-bef)
    cols={'xp':xp,'Rp':Rp,'bef':bef,'aft':aft,'sb':sb,'sa':sa,'mo':so[:,0],'m2o':so[:,1],
          'g*xp':g*xp,'g*mo':g*so[:,0],'g*m2o':g*so[:,1],'g2*xp':g2*xp,'g2*mo':g2*so[:,0],'g2*m2o':g2*so[:,1],'m1':m1,'m2':m2,'m3':m3,'x9':x[:,9],'M8':rm[:,8],'g*sa':g*sa,'g*sb':g*sb,
          'relu(xp-aft)':relu(xp-aft),'relu(xp-mo)':relu(xp-so[:,0]),'relu(xp-m2o)':relu(xp-so[:,1])}
    return list(cols), np.stack(list(cols.values()),1)
def pbank2(p):
    xp=x[:,p]; xo=np.delete(x,p,1); so=-np.sort(-xo,1); sx=xo.sum(1)
    g=(am==p).astype(float); g2=(a2==p).astype(float); n=1-g-g2
    base={'xp':xp,'so':sx,'mo':so[:,0],'m2o':so[:,1],'m3o':so[:,2]}
    cols={}
    for gn,gv in [('',1.0),('g*',g),('g2*',g2),('n*',n)]:
        for k,v in base.items(): cols[gn+k]=gv*v
    cols['r(xp-m2o)']=relu(xp-so[:,1]); cols['r(xp-mo)']=relu(xp-so[:,0]); cols['r(xp-m3o)']=relu(xp-so[:,2])
    cols['1']=np.ones(N)
    return list(cols), np.stack(list(cols.values()),1)
rank=np.argsort(o,1)
m4=s[:,3]
Q2=np.where(rank<=1,x,0.); Q1=np.where(rank==0,x,0.); Q3=np.where(rank<=2,x,0.)
names3=names2+['m4']+['Q2_%d'%j for j in range(10)]+['Q1_%d'%j for j in range(10)]+['Q3_%d'%j for j in range(10)]+['r(x%d-m3)'%j for j in range(10)]+['one']
B3=np.concatenate([B2,m4[:,None],Q2,Q1,Q3,relu(x-m3[:,None]),np.ones((N,1))],1)
def score(y,cols,mode):
    A=np.stack(cols,1)
    if mode=='relu': return relufit(y,A,it=5)
    return fit(y,cols)
def psearch(y,basefn,modfn,key=None,mode='lin',sub=40000,maxit=12):
    key=am if key is None else key
    yy=y[:sub]; kk=key[:sub]
    P=set(); best=score(yy,basefn(slice(0,sub))+modfn(np.isin(kk,list(P))*1.,slice(0,sub)),mode)[1]
    for _ in range(maxit):
        cand=None
        for q in range(10):
            P2=P^{q}
            r=score(yy,basefn(slice(0,sub))+modfn(np.isin(kk,list(P2))*1.,slice(0,sub)),mode)[1]
            if cand is None or r>cand[1]: cand=(P2,r)
        if cand[1]>best+0.002: P,best=cand
        else: break
    c,r=score(y,basefn(slice(None))+modfn(np.isin(key,list(P))*1.,slice(None)),mode)
    return sorted(P),c,r

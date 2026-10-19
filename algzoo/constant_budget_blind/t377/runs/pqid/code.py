import numpy as np, sys, json
import probe
rng=np.random.default_rng(7); x=rng.standard_normal((40000,10)); h=probe.hidden(x)
xt=np.random.default_rng(8).standard_normal((40000,10)); ht=probe.hidden(xt)
MODE=sys.argv[2]
def prep(x):
    if MODE in ('Q8','Q2'): x=x[:,:9]
    o=np.argsort(-x,1); v=-np.sort(-x,1); return o[:,:3],v[:,:3]
P,V=prep(x); Pt,Vt=prep(xt)
def design(P,V,u):
    if MODE=='Q2':
        g1=V[:,0]-V[:,1]; g2=V[:,1]-V[:,2]
        return np.c_[g1,g1*u[:10][P[:,0]],g2,g2*u[10:][P[:,1]],V[:,2]]
    if MODE=='R2':
        U1=u[:10][P[:,0]]; U2=u[10:][P[:,1:]]
        return np.c_[V[:,0],V[:,0]*U1,V[:,1]+V[:,2],(V[:,1:]*U2).sum(1),np.ones(len(V))]
    U=u[P]
    if MODE!='R':
        g1=V[:,0]-V[:,1]; g2=V[:,1]-V[:,2]
        return np.c_[g1,g1*U[:,0],g2,g2*U[:,1],V[:,2]]
    return np.c_[V[:,0],V[:,0]*U[:,0],V[:,1]+V[:,2],V[:,1]*U[:,1]+V[:,2]*U[:,2],np.ones(len(V))]
def fitc(X,y):
    g=y>0.3
    for it in range(6):
        c,*_=np.linalg.lstsq(X[g],y[g],rcond=None); g=(y>0.3)|(X@c>0)
    return c
def r2(c,X,y): return 1-((y-np.maximum(X@c,0))**2).mean()/y.var()
VALS=[-2,-1,0,1,2]
for i in json.loads(sys.argv[1]):
    y=h[:,i]
    # init from free per-position M weights
    oh=np.eye(10)
    Xf=np.c_[V[:,0:1]*oh[P[:,0]],np.ones(len(V))]
    cf=fitc(Xf,y); w=cf[:10]; w=(w-w.mean())/(w.std()+1e-9)
    u=np.clip(np.round(w*1.0),-2,2)
    if MODE=='Q8': u=np.r_[u[:9],0]
    if MODE=='R2': u=np.r_[u,u]
    if MODE=='Q2': u=np.r_[u[:9],0,u[:9],0]
    best=r2(fitc(design(P,V,u),y),design(P,V,u),y)
    for sweep in range(3):
        ch=False
        for j in ([k for k in range(20) if k not in (9,19)] if MODE=='Q2' else range(9 if MODE=='Q8' else (20 if MODE=='R2' else 10))):
            for val in VALS:
                if val==u[j]: continue
                u2=u.copy(); u2[j]=val; X=design(P,V,u2); s=r2(fitc(X,y),X,y)
                if s>best+1e-4: best,u,ch=s,u2,True
        if not ch: break
    c=fitc(design(P,V,u),y)
    print(i,'train %.4f test %.4f'%(best,r2(c,design(Pt,Vt,u),ht[:,i])),'u',u.astype(int).tolist(),'c',np.round(c,3).tolist(),flush=True)

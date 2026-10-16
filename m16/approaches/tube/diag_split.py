import sys; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/tube'); sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np, warnings; warnings.filterwarnings('ignore')
from tube import *
from arc_eval import Wi0, Wh0, Wo0, forward
M=(Wi0,Wh0,Wo0)
def kappa(x):
    """region and slope of each input: 0 blind, 1 R23, 2 R12; kappa = |gap|/|face point|"""
    o=np.argsort(-x,1); v=np.take_along_axis(x,o,1)
    reg=np.where(v[:,1]<=0,0,np.where(v[:,1]-v[:,2]<v[:,0]-v[:,1],1,2))
    s=np.where(reg==2,v[:,0]-v[:,1],v[:,1]-v[:,2])
    a=np.where(reg==2,o[:,0],o[:,1]); b=np.where(reg==2,o[:,1],o[:,2])
    x0=x.copy(); m=(x[np.arange(len(x)),a]+x[np.arange(len(x)),b])/2
    x0[np.arange(len(x)),a]=m; x0[np.arange(len(x)),b]=m
    return reg, s/np.linalg.norm(x0,axis=1)
x=np.random.default_rng(5).standard_normal((2_000_000,10))
reg,k=kappa(x); err=forward(M,x).argmax(1)!=np.argsort(x,1)[:,-2]
n=4096
for k0 in [0.02,0.04,0.08,0.16]:
    far=(reg>0)&(k>=k0); pf=(err&far).mean()
    out=[]
    for fn in (faces23,faces12):
        f=fn(n); f2=dict(f); rho=np.linalg.norm(f['x0'],axis=1); f2['hi']=np.minimum(f['hi'],k0*rho)
        F,_,C=F_faces(M,f2,400,radial=True); F*=f['Z']; out.append((F.mean(),F.std(),C.mean()))
    eff=(sum(sd*np.sqrt(c) for _,sd,c in out)+np.sqrt(pf*(1-pf))+0.0044)**2
    print('k0',k0,'far mass',far.mean(),'far err',pf,'near(23,12) mean/sd/cost',np.round(out,4).tolist(),'var*cost',eff,'gain vs sampling',0.0447/eff)

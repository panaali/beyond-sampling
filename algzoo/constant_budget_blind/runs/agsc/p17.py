import numpy as np, probe
from feat import r2; from fit import fitrelu
from lib import library
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(16); N=30000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
xt=rng.standard_normal((N,10)); ht=probe.hidden(xt)
def plib(x,p):
    L=library(x); N=len(x)
    pre=np.sort(x[:,:p],1)[:,::-1]; post=np.sort(x[:,p+1:],1)[:,::-1] if p<9 else np.full((N,2),-3.)
    if post.shape[1]<2: post=np.c_[post,np.full(N,-3.)]
    m2=pre[:,1] if p>1 else pre[:,0]; m3=pre[:,2] if p>2 else m2
    L.update({'pre1':pre[:,0],'pre2':m2,'pre3':m3,'xp':x[:,p],'Dp':np.maximum(0.4*pre[:,0]+0.6*m2-x[:,p],0),
      'Dp1':np.maximum(pre[:,0]-x[:,p],0),'post1':post[:,0],'post2':post[:,1],'postrelu':np.maximum(x[:,p+1:],0).sum(1) if p<9 else np.zeros(N),
      'pre12':pre[:,0]+m2,'cap':np.minimum(x[:,p],pre[:,0])})
    return L
for i,ps in [(0,[4,5]),(1,[6]),(2,[3,5]),(3,[5,7]),(6,[4]),(7,[5,6])]:
    for p in ps:
        L=plib(x,p); Lt=plib(xt,p); keys=list(L); y=h[:,i]; sel=[]; one=np.ones(N)
        for k in range(4):
            best=None
            for key in keys:
                if key in sel: continue
                A=np.c_[np.array([L[q] for q in sel+[key]]).T,one]
                c=np.linalg.lstsq(A,y,rcond=None)[0]; r=r2(y,A@c)
                if best is None or r>best[0]: best=(r,key)
            sel.append(best[1])
        A=np.c_[np.array([L[q] for q in sel]).T,one]; c,rr=fitrelu(A,y)
        At=np.c_[np.array([Lt[q] for q in sel]).T,one]
        print(i,'p',p,sel,'test',round(r2(ht[:,i],np.maximum(At@c,0)),3),np.round(c,2))

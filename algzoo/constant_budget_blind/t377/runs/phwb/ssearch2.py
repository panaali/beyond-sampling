import numpy as np, sys
from lib import relufit
X=np.load('x0.npy');H=np.load('h0.npy')
def base(x):
    o=np.argsort(-x,1); s=-np.sort(-x,1); return o[:,0],o[:,1],s[:,0],s[:,1],s[:,2]
x=X[:15000];h=H[:15000];xt=X[80000:100000];ht=H[80000:100000]
B=base(x);Bt=base(xt)
def des(B,S,form):
    p1,p2,m1,m2,m3=B; i1=np.isin(p1,S).astype(float); i2=np.isin(p2,S).astype(float); one=np.ones(len(p1))
    if form==1: return np.c_[m1*i1,m2*i2,m1,one]
    if form==2: return np.c_[m1*i1,m2*i2,m1,m2,one]
    if form==3: return np.c_[(m1-m3)*i1,(m2-m3)*i2,m1,one]
subs=[[k for k in range(10) if m>>k&1] for m in range(1,1023)]
def ls(A,y,g):
    c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None); return 1-((A[g]@c-y[g])**2).mean()/y[g].var()
for i in [int(a) for a in sys.argv[1].split(',')]:
    y=h[:,i];g=y>0.3
    for form in (1,2,3):
        r,S=max((ls(des(B,S,form),y,g),S) for S in subs)
        c,_=relufit(des(B,S,form),y); pt=np.maximum(des(Bt,S,form)@c,0)
        print(i,form,'S=',S,np.round(c,2),'R2test',round(1-((pt-ht[:,i])**2).mean()/ht[:,i].var(),3),flush=True)

import numpy as np, probe, sys
from gate import fastfit, lmfit
from lib import lib
np.set_printoptions(precision=2,suppress=True,linewidth=200)
rng=np.random.default_rng(12)
x=rng.standard_normal((20000,10)); H=probe.hidden(x); F=lib(x)
s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
srt=-np.sort(-x,1)
def run(i,k):
    h=H[:,i]
    rec=np.ones(len(x),bool) if k==0 else x[:,k]>x[:,:k].max(1)
    top2=x[:,k]>=F['so%d'%k]
    out=[]
    for gn,g in [('all',np.ones(len(x),bool)),('rec',rec),('top2',top2),('am',am==k),('rec|top2',rec|top2)]:
        X=np.c_[x,srt[:,:3]]*g[:,None]
        c,r=fastfit(h,X); out.append("%s %.3f (in-gate mass %.2f)"%(gn,r,h[g].sum()/h.sum()))
    print(i,k," | ".join(out))
    g=rec
    X=np.c_[x,srt[:,:3]]*g[:,None]; c,r=fastfit(h,X); print("   rec coefs x0..9,T1,T2,T3",c)
for a in sys.argv[1:]:
    i,k=map(int,a.split(':')); run(i,k)

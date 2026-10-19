import numpy as np, probe, sys
from gate import fastfit, lmfit
np.set_printoptions(precision=2,suppress=True,linewidth=200)
rng=np.random.default_rng(11)
x=rng.standard_normal((20000,10)); H=probe.hidden(x)
p=x[:,:9]; s=np.argsort(-p,1); am9=s[:,0]; a29=s[:,1]; srt=-np.sort(-p,1); M9=srt[:,0]; S9=srt[:,1]; x9=x[:,9]
E=np.eye(9)
X=np.c_[M9[:,None]*E[am9], S9[:,None]*E[a29], x9]
for i in [int(a) for a in sys.argv[1:]]:
    h=H[:,i]; c,r=fastfit(h,X); c=lmfit(h,X,c); pr=np.maximum(X@c,0); r=1-((pr-h)**2).mean()/h.var()
    print(i,"R2 %.3f"%r); print("  M9 by am9",c[:9]); print("  S9 by a2_9",c[9:18]); print("  x9",c[18])

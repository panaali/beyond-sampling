import numpy as np, probe, sys
from gate import fastfit, lmfit
np.set_printoptions(precision=2,suppress=True,linewidth=200)
rng=np.random.default_rng(13)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
s=np.argsort(-x,1); srt=-np.sort(-x,1); E=np.eye(10)
X=np.c_[srt[:,0:1]*E[s[:,0]], srt[:,1:2]*E[s[:,1]], srt[:,2:3]*E[s[:,2]], srt[:,3:4]*E[s[:,3]]]
for i in [int(a) for a in sys.argv[1:]]:
    h=H[:,i]; c,r=fastfit(h,X); c=lmfit(h,X,c); pr=np.maximum(X@c,0); r=1-((pr-h)**2).mean()/h.var()
    r3=fastfit(h,X[:,:30])[1]; r2=fastfit(h,X[:,:20])[1]
    print(i,"R2 top2 %.3f top3 %.3f top4 %.3f"%(r2,r3,r)); 
    for j,nm in enumerate(['T1@am','T2@a2','T3@a3','T4@a4']): print("  ",nm,c[10*j:10*j+10])

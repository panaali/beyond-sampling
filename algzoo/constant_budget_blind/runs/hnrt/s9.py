import numpy as np, probe
from util import *
rng=np.random.default_rng(8)
x=rng.standard_normal((60000,10)); H=probe.hidden(x); h=H[:,2]
rec2=x[:,2]>x[:,:2].max(1); gm=x.argmax(1)==2
o=np.argsort(-x,1); r2nd=o[:,1]==2
for nm,m in [('rec2&gm',rec2&gm),('rec2&~gm',rec2&~gm),('~rec2',~rec2),('2nd==2',r2nd),('rec2&2nd',rec2&r2nd)]:
    print(nm,m.mean().round(3),h[m].mean().round(2),(h[m]>0).mean().round(2), np.corrcoef(h[m],x[m,2])[0,1].round(2))
m=rec2&~gm
# within rec2 & not global max: regress h on x2 and later max Q3, and x1,x0
for feats in [['x2'],['x2','Q3'],['x2','Q3','x0x1max']]:
    cols={'x2':x[:,2],'Q3':x[:,3:].max(1),'x0x1max':x[:,:2].max(1)}
    F=np.stack([cols[c][m] for c in feats],1); w=relufit(F,h[m]); print(feats,np.round(w,2),round(r2(np.maximum(F@w,0),h[m]),3))
# what fraction of variance of h lies in rec2&~gm
print("var share", ((h[m]-h.mean())**2).sum()/((h-h.mean())**2).sum())

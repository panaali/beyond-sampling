import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
H=probe.hidden(x); p=probe.predict(x)
o=np.argsort(-x,1); a1=o[:,0]; a2=o[:,1]
print("acc",(p==a2).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print("mean",H.mean(0)); print("std",H.std(0)); print("frac>0",(H>0).mean(0))
s=-np.sort(-x,1)
feats={'x%d'%j:x[:,j] for j in range(10)}
feats.update({'s1':s[:,0],'s2':s[:,1],'s3':s[:,2],'s1-s2':s[:,0]-s[:,1],'s2-s3':s[:,1]-s[:,2]})
for i in range(16):
    c={k:np.corrcoef(v,H[:,i])[0,1] for k,v in feats.items()}
    top=sorted(c.items(),key=lambda t:-abs(t[1]))[:6]
    # mean by argmax and second position
    m1=[H[a1==j,i].mean() for j in range(10)]; m2=[H[a2==j,i].mean() for j in range(10)]
    print(i,[(k,round(v,2)) for k,v in top]); print('  byA1',np.array(m1)); print('  byA2',np.array(m2))

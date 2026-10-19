import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
H=probe.hidden(x); p=probe.predict(x)
np.save('x1.npy',x); np.save('H1.npy',H)
o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
print('acc',(p==a2).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
s=-np.sort(-x,1)
for i in range(32):
    h=H[:,i]
    cx=[np.corrcoef(h,x[:,t])[0,1] for t in range(10)]
    cs=[np.corrcoef(h,s[:,k])[0,1] for k in range(4)]
    mam=[h[am==t].mean() for t in range(10)]
    ma2=[h[a2==t].mean() for t in range(10)]
    print(i,'mean %.2f act %.2f'%(h.mean(),(h>1e-6).mean()))
    print('  cx',np.array(cx),' cs',np.array(cs))
    print('  m|am',np.array(mam)); print('  m|a2',np.array(ma2))

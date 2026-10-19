import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
H=probe.hidden(x); np.save('x1.npy',x); np.save('H1.npy',H)
p=probe.predict(x); sec=np.argsort(-x,1)[:,1]; print('acc',(p==sec).mean())
s=-np.sort(-x,1); am=np.argmax(x,1)
def r2(F,y):
    F=np.c_[F,np.ones(len(y))]; c,*_=np.linalg.lstsq(F,y,rcond=None); return 1-((F@c-y)**2).mean()/y.var()
for i in range(32):
    y=H[:,i]
    print(i,'mean %.2f frac0 %.2f'%(y.mean(),(y<1e-6).mean()),'R2 x %.2f sorted %.2f x+sorted %.2f'%(r2(x,y),r2(s,y),r2(np.c_[x,s],y)),
      'corr w x:',np.round([np.corrcoef(x[:,j],y)[0,1] for j in range(10)],2))

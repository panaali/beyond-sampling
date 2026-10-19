import numpy as np, probe
from lib2 import fitrelu
rng=np.random.default_rng(31); x=rng.standard_normal((20000,10)).astype(np.float32)
# h_t via zero prefix: shift sequence right by (10-t)
for i,j in [(24,5),(30,4),(20,0)]:
    print('neuron',i)
    for t in range(j+1,11):
        z=np.zeros_like(x); z[:,10-t:]=x[:,:t]; hh=probe.hidden(z)[:,i]; xs=x[:,:t].astype(float)
        m=np.maximum.accumulate(xs,1); s=np.sort(xs,1)[:,::-1]
        o=np.delete(xs,j,1); others=np.sort(o,1)[:,::-1]
        X=np.c_[xs[:,j],others[:,0],others[:,1] if t>2 else others[:,0], m[:,j-1] if j>0 else xs[:,j]*0, xs[:,t-1]]
        r2,c=fitrelu(X,hh,10)
        print('  t=%d mean %.2f frac %.2f R2 %.3f coef[xj,max_others,2nd_others,runmax_before_j,x_last]'%(t,hh.mean(),(hh>0).mean(),r2),np.round(c,2))

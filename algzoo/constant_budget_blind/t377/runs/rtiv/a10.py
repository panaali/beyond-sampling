import numpy as np
d=np.load('d1.npz');x=d['x'];H=d['H']
o=np.argsort(-x,1);am=o[:,0];a2=o[:,1];M=x.max(1);S=-np.sort(-x,1)[:,1]
ev=(am%2==0)
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in range(16):
    y=H[:,i];pred=np.zeros_like(y);cs=[]
    for g in [ev,~ev]:
        X=np.c_[M[g],S[g],x[g]];c,*_=np.linalg.lstsq(X,y[g],rcond=None);pred[g]=X@c;cs.append(c)
    r=1-((y-pred)**2).mean()/y.var()
    pr=np.maximum(pred,0); r_relu=1-((y-pr)**2).mean()/y.var()
    print("%2d r2=%.3f"%(i,r),"even:",cs[0][:2],cs[0][2:],"\n         odd :",cs[1][:2],cs[1][2:])

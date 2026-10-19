import numpy as np
d=np.load('d1.npz');x=d['x'];H=d['H']
o=np.argsort(-x,1);am=o[:,0];M=x.max(1);S=-np.sort(-x,1)[:,1]
T=np.arange(10)[None]
prev=np.where(T<am[:,None],x,-np.inf).max(1)  # max before argmax
np.set_printoptions(precision=2,suppress=True,linewidth=220)
for i,t in [(6,9),(8,7),(12,9)]:
  print("neuron",i)
  for k in range(1,10):
    m=am==k; y=H[m,i]; X=np.c_[M[m],x[m,t],prev[m],S[m]]
    c,*_=np.linalg.lstsq(X,y,rcond=None); r=1-((y-X@c)**2).mean()/y.var()
    X2=np.c_[M[m],x[m]];c2,*_=np.linalg.lstsq(X2,y,rcond=None); r2=1-((y-X2@c2)**2).mean()/y.var()
    print(k,"M,xt,prev,S",c,round(r,3)," | M+x:",round(r2,3),c2)

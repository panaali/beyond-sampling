import numpy as np, probe, itertools
from feat import r2; from fit import fitrelu
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(14); N=60000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
xt=rng.standard_normal((N,10)); ht=probe.hidden(xt)
def parts(x):
    o=np.argsort(-x,1); r=np.argsort(o,1)
    return x*(r<2), x*(r>=2), x*(r==2)
E,O,T3=parts(x); Et,Ot,T3t=parts(xt); one=np.ones(N)
for i in [0,1,2,3,6,7]:
    y=h[:,i]; A=np.c_[E,O,one]; c=np.linalg.lstsq(A,y,rcond=None)[0]
    we,wo=c[:10],c[10:20]
    A=np.c_[E,O,one]; cr,rr=fitrelu(A,y); print(i,'relu full21 train R2',round(rr,3))
    idx=np.argsort(we); out=[]
    for a in range(1,9):
        for b in range(a+1,10):
            cls=np.zeros(10,int); cls[idx[a:b]]=1; cls[idx[b:]]=2
            for thr in [0.4,0.6,0.9]:
              Bn=np.where(wo<-thr)[0]; Bp=np.where(wo>thr)[0]
              def F(E,O):
                  f=[E[:,cls==k].sum(1) for k in range(3)]+[O[:,Bn].sum(1)]
                  f.append(O[:,Bp].sum(1) if len(Bp) else np.ones(len(E)))
                  return np.array(f).T
              A=F(E,O); cc=np.linalg.lstsq(A,y,rcond=None)[0]; out.append((r2(y,A@cc),a,b,thr,cls.copy(),Bn,Bp))
    out.sort(key=lambda z:-z[0]); r0,a,b,thr,cls,Bn,Bp=out[0]
    def F(E,O):
        f=[E[:,cls==k].sum(1) for k in range(3)]+[O[:,Bn].sum(1)]
        f.append(O[:,Bp].sum(1) if len(Bp) else np.ones(len(E))); return np.array(f).T
    cc,_=fitrelu(F(E,O),y)
    print('  cls',cls,'neg',Bn,'pos',Bp,'test',round(r2(ht[:,i],np.maximum(F(Et,Ot)@cc,0)),3),np.round(cc,2))

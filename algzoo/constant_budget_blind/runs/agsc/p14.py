import numpy as np, probe, itertools
from feat import r2; from fit import fitrelu
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(13); N=60000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
xt=rng.standard_normal((N,10)); ht=probe.hidden(xt)
def parts(x):
    o=np.argsort(-x,1); r=np.argsort(o,1)
    return x*(r<2), x*(r>=2)
E,O=parts(x); Et,Ot=parts(xt); one=np.ones(N)
for i in [0,1,2,3,6,7]:
    y=h[:,i]; A=np.c_[E,O,one]; c=np.linalg.lstsq(A,y,rcond=None)[0]
    we,wo=c[:10],c[10:20]
    print(i,'full20 R2',round(r2(y,A@c),3),'top2 prof',we,'oth prof',wo)
    best=None
    # 3-level partition of positions for top2 weights: search by sorted thresholds of we
    idx=np.argsort(we)
    for a in range(1,9):
        for b in range(a+1,10):
            cls=np.zeros(10,int); cls[idx[a:b]]=1; cls[idx[b:]]=2
            for oth in ['neg','none']:
                Bset=np.where(wo<-0.6)[0] if oth=='neg' else []
                F=[E[:,cls==k].sum(1) for k in range(3)]
                Ft=[Et[:,cls==k].sum(1) for k in range(3)]
                if len(Bset): F.append(O[:,Bset].sum(1)); Ft.append(Ot[:,Bset].sum(1))
                A=np.c_[np.array(F).T,one]; cc=np.linalg.lstsq(A,y,rcond=None)[0]; rr=r2(y,A@cc)
                if best is None or rr>best[0]: best=(rr,cls.copy(),list(Bset),oth)
    rr,cls,Bset,oth=best
    F=[E[:,cls==k].sum(1) for k in range(3)]+([O[:,Bset].sum(1)] if Bset else [])
    Ft=[Et[:,cls==k].sum(1) for k in range(3)]+([Ot[:,Bset].sum(1)] if Bset else [])
    A=np.c_[np.array(F).T,one]; cc,_=fitrelu(A,y); At=np.c_[np.array(Ft).T,one]
    print('   best cls',cls,'Bset',Bset,'test R2',round(r2(ht[:,i],np.maximum(At@cc,0)),3),'consts',np.round(cc,2))

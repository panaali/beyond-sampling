import numpy as np, probe
from feat import r2; from fit import fitrelu
from lib2 import lib3
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(21); N=40000
x=rng.standard_normal((N,10)); h=probe.hidden(x); L=lib3(x); one=np.ones(N)
xt=rng.standard_normal((60000,10)); ht=probe.hidden(xt); Lt=lib3(xt); onet=np.ones(len(xt))
for fam in ['DR','DA']:
  ts=range(1,10) if fam=='DR' else range(2,10)
  for G in ['M+S','m1_9']:
    D=np.array([L[f'{fam}{t}'] for t in ts]).T; Dt=np.array([Lt[f'{fam}{t}'] for t in ts]).T
    if fam=='DA': D=np.c_[L['DR1'],D]; Dt=np.c_[Lt['DR1'],Dt]
    for i in [0,1,2,3,6,7]:
        y=h[:,i]; A=np.c_[D,L[G],one]; c,rr=fitrelu(A,y)
        w=c[:9]; idx=np.argsort(w); best=None
        for a in range(1,8):
            for b in range(a+1,9):
                cls=np.zeros(9,int); cls[idx[a:b]]=1; cls[idx[b:]]=2
                F=np.c_[[D[:,cls==k].sum(1) for k in range(3)]].T
                A2=np.c_[F,L[G],one]; cc=np.linalg.lstsq(A2,y,rcond=None)[0]; r=r2(y,A2@cc)
                if best is None or r>best[0]: best=(r,cls)
        cls=best[1]; F=np.c_[[D[:,cls==k].sum(1) for k in range(3)]].T; Ft=np.c_[[Dt[:,cls==k].sum(1) for k in range(3)]].T
        cc,_=fitrelu(np.c_[F,L[G],one],y)
        rt=r2(ht[:,i],np.maximum(np.c_[Ft,Lt[G],onet]@cc,0))
        print(fam,G,i,'full',round(rr,3),'w',w,'3cls test',round(rt,3),'cls(t=1..9)',cls,np.round(cc,2))

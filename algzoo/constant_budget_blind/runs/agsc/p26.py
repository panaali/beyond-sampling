import numpy as np, probe, labels
from feat import r2; from fit import fitrelu
rng=np.random.default_rng(35); x=rng.standard_normal((40000,10)); h=probe.hidden(x)
# ablation: replace drop features by raw -x_t or relu(-x_t)
abl={0:{0:[4,5],1:[3]},1:{0:[6],1:[4]},3:{1:[3],2:[5],3:[7]},4:{0:[8]},6:{1:[4],2:[2]},7:{1:[5],2:[6]}}
for i,cols in abl.items():
    A=labels._design(i,x); _,r0=fitrelu(A,h[:,i]); B=A.copy(); B2=A.copy()
    for j,ts in cols.items():
        B[:,j]=-x[:,ts].sum(1); B2[:,j]=np.maximum(-x[:,ts],0).sum(1)
    _,r1=fitrelu(B,h[:,i]); _,r2_=fitrelu(B2,h[:,i])
    print(i,'drop-below-max',round(r0,3),'raw -x_t',round(r1,3),'relu(-x_t)',round(r2_,3))

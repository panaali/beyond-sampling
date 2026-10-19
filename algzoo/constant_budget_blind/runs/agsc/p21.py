import numpy as np, probe
from feat import r2; from fit import fitrelu
from lib2 import lib3
rng=np.random.default_rng(20); N=40000
x=rng.standard_normal((N,10)); h=probe.hidden(x); L=lib3(x); one=np.ones(N)
L['DR45']=L['DR4']+L['DR5']
base={0:['DR45','DR3','x8','M+S'],1:['DA6','DA4','m1_9','MSlag'],2:['m1_9','DA3','DA5','DA6'],3:['m1_9','DR3','DR5','DA7'],6:['M+S','DR4','DR2','x8'],7:['MSlag','DA5','DR6','x9']}
keys=list(L)
for k in list(keys):
    if k.startswith('DR') or k.startswith('DA'):
        L[k+'*M']=L[k]*L['M']; L[k+'^2']=L[k]**2
keys=list(L)
for i,sel in base.items():
    A=np.c_[np.array([L[q] for q in sel]).T,one]; c,rr=fitrelu(A,h[:,i]); p=np.maximum(A@c,0); res=h[:,i]-p
    gains=[]
    for k in keys:
        if k in sel: continue
        A2=np.c_[A,L[k]]; c2,r2b=fitrelu(A2,h[:,i],np.r_[c,0]); gains.append((round(r2b-rr,3),k))
    gains.sort(reverse=True); print(i,round(rr,3),gains[:6])

"""Diagnosis only: Monte Carlo decomposition of the error into blind / R23 / R12 regions."""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np, json
from arc_eval import Wi0, Wh0, Wo0, forward
M=(Wi0,Wh0,Wo0); rng=np.random.default_rng(777); acc=np.zeros(3); N=0
for _ in range(8):
    x=rng.standard_normal((500000,10)); v=-np.sort(-x,1)
    err=forward(M,x).argmax(1)!=np.argsort(x,1)[:,-2]
    b=v[:,1]<=0; r23=(v[:,1]-v[:,2]<v[:,0]-v[:,1])&~b; r12=~b&~r23
    acc+=[(err&b).sum(),(err&r23).sum(),(err&r12).sum()]; N+=len(x)
print(json.dumps(dict(blind=acc[0]/N,e23=acc[1]/N,e12=acc[2]/N,se=[np.sqrt(a)/N for a in acc])))

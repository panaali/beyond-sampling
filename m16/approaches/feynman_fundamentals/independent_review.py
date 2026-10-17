"""Reproduction of T47 agent's independently designed/read-only T55 checks.

Saved by T55 from the reviewer's exact vectors, formulas, and RNG seed.
The recorded review was run by T47, not rerun by T55 when saving this file.
"""
import json
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma,ndtr
from tail import conditional_probability,prefix_summary

start=time.process_time()
rows=[]
prefixes=([2.,-1.],[-.5,2.,-1.],[0.,3.,2.,-1.],
          [8.,1.,-2.,3.,-4.,5.,-6.,7.],[-1.,-2.,-3.],[1.])
for values in prefixes:
    prefix=np.array(values);k=len(prefix)
    model=(np.array([.8,-.6]),np.array([[.35,-.7],[.2,.45]]),np.ones((k+2,2)))
    probability,_=conditional_probability(model,*prefix_summary(model,prefix),k)
    u=prefix/np.linalg.norm(prefix)
    rank=int(np.sum(u>u[0]))
    def integrand(radius):
        c=ndtr(radius*u[0])
        mass=2*c*(1-c) if rank==0 else c*c if rank==1 else 0.
        return mass*radius**(k-1)*np.exp(-radius*radius/2)/(2**(k/2-1)*gamma(k/2))
    reference,error=quad(integrand,0,np.inf,epsabs=1e-12)
    rows.append(dict(prefix=values,probability=probability,reference=reference,
                     difference=probability-reference,quad_error=error))

rng=np.random.default_rng(552204)
prefix=np.array([-.9,.4,1.2]);k=3;T=5
wo=-np.ones((T,1));wo[1:3]=1
model=(np.ones(1),np.zeros((1,1)),wo)
probability,_=conditional_probability(model,*prefix_summary(model,prefix),k)
radius=np.sqrt(rng.chisquare(k,262144))
x=np.column_stack([radius[:,None]*prefix/np.linalg.norm(prefix),rng.normal(size=(len(radius),2))])
prediction=np.where(x[:,-1]>0,1,0)
hit=prediction==np.argsort(x,axis=1)[:,-2]
mc=float(hit.mean());se=float(np.sqrt(mc*(1-mc)/len(radius)))
print(json.dumps(dict(constant_readouts=rows,tie_case=dict(probability=probability,
    mc=mc,se=se,z=(probability-mc)/se),cpu_seconds=time.process_time()-start),indent=2))

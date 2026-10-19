import numpy as np, json, sys
from featlib import *
from feats import fit_relu
x=np.load('x1.npy')[:40000];H=np.load('H1.npy')[:40000]
def ev(e,x): return eval(e)
def fit(kind,feats,gate,y,x):
    F=np.column_stack([ev(e,x) for e in feats]) if feats else np.zeros((len(y),0))
    if kind=='lin':
        Xb=np.hstack([F,np.ones((len(y),1))]); c,*_=np.linalg.lstsq(Xb,y,rcond=None); return c
    if kind=='scale':  # c*feat0, no bias
        f0=F[:,0]; return np.array([ (f0@y)/(f0@f0) ])
    g=np.ones(len(y),bool) if gate is None else ev(gate,x)>0.5
    r,c=fit_relu(F[g],y[g],25); return c
def predict(kind,feats,gate,c,x):
    F=np.column_stack([ev(e,x) for e in feats])
    if kind=='scale': return c[0]*F[:,0]
    z=F@c[:-1]+c[-1]
    if kind=='lin': return z
    p=relu(z)
    if gate is not None: p=p*(ev(gate,x)>0.5)
    return p
SPEC=json.load(open('spec.json'))
out={}
for k,(kind,feats,gate) in SPEC.items():
    i=int(k); c=fit(kind,feats,gate,H[:,i],x); out[k]=[kind,feats,gate,[float(v) for v in c]]
json.dump(out,open('fitted.json','w'),indent=0)
rng=np.random.default_rng(11); import probe
xt=rng.standard_normal((40000,10)); Ht=np.asarray(probe.hidden(xt))
for k,(kind,feats,gate,c) in out.items():
    i=int(k); p=predict(kind,feats,gate,np.array(c),xt); y=Ht[:,i]
    print(i,'R2test %.3f'%(1-((y-p)**2).mean()/y.var()),'nc',len(c),np.round(c,3))

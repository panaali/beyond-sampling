import numpy as np, probe
from fitlib import *
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
xt=np.random.default_rng(7).standard_normal((30000,10));ht=probe.hidden(xt)
def D(x):
    w=-np.sort(-x[:,:5],1);F=pf(x)
    base=[w[:,1],np.maximum(x[:,2],0),F['S']]
    ex={'w0':w[:,0],'w2':w[:,2],'r4':np.maximum(x[:,4],0),'r0':np.maximum(x[:,0],0),'T':F['T'],'M':F['M'],'late':x[:,5:9].max(1),'x9':x[:,9]}
    return base,ex
b,e=D(x);bt,et=D(xt)
for k in e:
    X=np.c_[b+[e[k]]].T;Xt=np.c_[bt+[et[k]]].T
    c,*_=np.linalg.lstsq(X,h[:,2],rcond=None)
    r=least_squares(lambda th:np.maximum(0,X@th)-h[:,2],c)
    print(k,'%.3f'%r2(np.maximum(0,Xt@r.x),ht[:,2]),np.round(r.x,2))

import sys, numpy as np
sys.path.insert(0,"/home/user/clean-neurons/m16")
from arc_eval import noisy_models
from common import Wi,Wh,Wo
import cv
from scipy.stats import norm
rng=np.random.default_rng(3)
def feats(x):
    s=-np.sort(-x[:,:9],1); m1,m2,m3=s[:,0],s[:,1],s[:,2]
    f=[norm.pdf(m1)*np.abs(m1),norm.pdf(m2)*np.abs(m2),norm.pdf(m1),norm.pdf(m2),norm.cdf(m2),norm.cdf(m1),m2/np.maximum(m1,1e-9),(m2-m3)/np.maximum(np.abs(m2),1e-9)]
    return np.stack(f,1)
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1])]:
    for h in (4,5,6):
        x=cv.draw(h,20000,rng); q=cv.Q(m,x)
        for nm,B in (("A",cv.basis9(x,"A")),("phi",feats(x)),("A+phi",np.hstack([cv.basis9(x,"A"),feats(x)]))):
            X=np.hstack([np.ones((len(x),1)),B]); hh=10000
            c=np.linalg.lstsq(X[:hh],q[:hh],rcond=None)[0]; r=q[hh:]-X[hh:]@c
            print(name,h,nm,f"var ratio {q[hh:].var()/r.var():.2f}")

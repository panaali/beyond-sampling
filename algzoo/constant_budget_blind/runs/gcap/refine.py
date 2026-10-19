import numpy as np
from scipy.optimize import least_squares
import labels_core as L
x=np.load('x.npy')[:50000];H=np.load('H.npy')[:50000]
for i in range(8):
    names,c0=L.SPEC[i]; F=np.c_[[L.feat(x,n) for n in names]].T; y=H[:,i]
    fun=lambda c: np.maximum(F@c,0)-y
    r=least_squares(fun,np.array(c0),method='lm' if False else 'trf')
    R0=1-np.mean(fun(np.array(c0))**2)/y.var(); R1=1-np.mean(r.fun**2)/y.var()
    print(i,round(R0,4),round(R1,4),r.x.round(4))

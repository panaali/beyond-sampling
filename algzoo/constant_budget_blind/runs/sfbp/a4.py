import numpy as np, probe
from plib import plib, greedy
rng=np.random.default_rng(8)
N=15000
x=rng.standard_normal((N,10))
def H(x9):
    z=x.copy(); z[:,9]=x9; return probe.hidden(z)
Hp,Hn,H0=H(40.),H(-40.),H(0.)
Hp2,Hn2=H(50.),H(-50.)
np.savez('d4.npz',x=x,Hp=Hp,Hn=Hn,H0=H0,Hp2=Hp2,Hn2=Hn2)
F=plib(x)
for i in range(32):
    sp=(Hp2[:,i]-Hp[:,i])/10; sn=(Hn2[:,i]-Hn[:,i])/-10
    if np.median(np.abs(sp))>0.5: w=np.median(sp); c=Hp[:,i]-40*w; m=np.abs(sp-w)<1e-3
    elif np.median(np.abs(sn))>0.1: w=np.median(sn); c=Hn[:,i]+40*w; m=np.abs(sn-w)<1e-3
    else:
        w=0; c=H0[:,i]; m=c>1e-6
    s,coef=greedy(F,c,4,m)
    print(i,'w=%.3f'%w,'act',m.mean().round(2),s)

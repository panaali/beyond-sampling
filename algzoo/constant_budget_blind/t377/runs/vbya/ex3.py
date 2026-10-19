import numpy as np, probe
np.set_printoptions(precision=3,suppress=True,linewidth=200)
rng=np.random.default_rng(2)
x=rng.standard_normal((2000,10))
out={}
for v in [-30,-20,20,30]:
    y=x.copy(); y[:,9]=v; out[v]=probe.hidden(y)
sn=(out[-20]-out[-30])/10   # = -w for active
sp=(out[30]-out[20])/10
print('slope neg side (=-w):',np.median(sn,0))
print('slope pos side (=w):',np.median(sp,0))
print('frac active at -30',(out[-30]>0).mean(0))
print('frac active at +30',(out[30]>0).mean(0))

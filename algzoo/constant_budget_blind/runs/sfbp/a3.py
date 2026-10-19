import numpy as np, probe
rng=np.random.default_rng(7)
N=20000
x=rng.standard_normal((N,10))
def H(x9):
    z=x.copy(); z[:,9]=x9; return probe.hidden(z)
T=[-30,-20,20,30]
Hs={t:H(t) for t in T}
w=np.zeros(32)
for i in range(32):
    sp=(Hs[30][:,i]-Hs[20][:,i])/10; sn=(Hs[-30][:,i]-Hs[-20][:,i])/-10
    # active direction
    a=sp[np.abs(sp)>1e-6]; b=sn[np.abs(sn)>1e-6]
    print(i,'pos-slope',np.round(np.median(a),3) if len(a) else None,len(a),'neg-slope',np.round(np.median(b),3) if len(b) else None,len(b))

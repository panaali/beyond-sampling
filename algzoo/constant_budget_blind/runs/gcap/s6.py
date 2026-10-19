import numpy as np, probe
x=np.load('x.npy')[:50000].copy()
X0=x.copy(); X0[:,9]=0; h0=probe.hidden(X0)
Xn=x.copy(); Xn[:,9]=-10; hn=probe.hidden(Xn)
Xp=x.copy(); Xp[:,9]=10; hp=probe.hidden(Xp)
w=(hp-hn)/20  # valid where both active... estimate per neuron by median of (hp-h0)/10 or (h0-hn)/10
print('w est from active', [np.median(((hp-h0)/10)[:,i][(h0[:,i]>0)&(hp[:,i]>0)]) if ((h0[:,i]>0)&(hp[:,i]>0)).sum()>0 else np.median(((h0-hn)/10)[:,i][(h0[:,i]>0)&(hn[:,i]>0)]) for i in range(8)])
np.save('h0.npy',h0);np.save('hn.npy',hn);np.save('hp.npy',hp)
print('frac h0>0',(h0>0).mean(0))

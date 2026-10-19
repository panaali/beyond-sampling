import numpy as np, probe
np.set_printoptions(precision=1,suppress=True,linewidth=220)
rng=np.random.default_rng(5)
base=rng.standard_normal((6,10))
vals=np.linspace(-3,4,15)
for i,t in [(21,1),(23,6),(29,0)]:
  print('neuron',i,'vary x%d'%t,'vals',vals)
  for b in base:
    X=np.tile(b,(len(vals),1)); X[:,t]=vals
    h=probe.hidden(X)[:,i]
    srt=-np.sort(-np.delete(b,t))
    print(' others sorted top3',srt[:3],' h:',h)

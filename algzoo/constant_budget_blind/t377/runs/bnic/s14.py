import numpy as np, probe
rng=np.random.default_rng(5); x=rng.standard_normal((20000,10))
H=[]
for t in range(1,11):
    xp=np.zeros_like(x); xp[:,10-t:]=x[:,:t]; H.append(probe.hidden(xp))
H=np.stack(H,1)  # N, t(1..10), 16
np.save('x14.npy',x); np.save('H14.npy',H)
np.set_printoptions(linewidth=200,suppress=True)
print('frac active per step (rows=neuron, cols t=1..10)')
print((H>0).mean(0).T.round(2))
print('corr h_t[i] with h_10[i]')
print(np.array([[np.corrcoef(H[:,t,i],H[:,9,i])[0,1] for t in range(10)] for i in range(16)]).round(2))

import numpy as np, probe
np.set_printoptions(linewidth=200,suppress=True)
x=np.random.default_rng(9).standard_normal((200000,10)); h=probe.hidden(x)[:,14]
am=np.argmax(x,1); m=am==6; x=x[m]; y=h[m]; s=np.sort(x,1)[:,::-1]
bins=np.linspace(-2.5,2.5,11)
for t in [6,0,3,5,7,8,9]:
    idx=np.digitize(x[:,t],bins); print('x%d'%t,[round(y[idx==k].mean(),1) if (idx==k).sum()>50 else None for k in range(12)])
# ratio y/x6
q=y/x[:,6]; print('y/x6 corr with x_t:',[np.corrcoef(q,x[:,t])[0,1].round(2) for t in range(10)])

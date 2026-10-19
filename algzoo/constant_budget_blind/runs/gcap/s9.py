import numpy as np, probe
np.set_printoptions(precision=2,suppress=True,linewidth=160)
x=np.load('x.npy')[:20000]
hs=[]
for k in range(1,11):
    X=np.zeros_like(x); X[:,10-k:]=x[:,:k]; hs.append(probe.hidden(X))
hs=np.array(hs) # hs[k-1]=h_k
np.save('hs.npy',hs)
H=hs[9]
for i in range(8):
    C=np.array([[np.corrcoef(hs[k][:,j],H[:,i])[0,1] if hs[k][:,j].std()>0 else 0 for j in range(8)] for k in range(5,9)])
    print('h10 neuron',i,'corr with h6..h9 neurons (rows=k)'); print(C)

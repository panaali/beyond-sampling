import numpy as np
x=np.load('x0.npy')[:50000];h=np.load('h0.npy')[:50000]
pm=np.maximum.accumulate(x,1); sm=np.maximum.accumulate(x[:,::-1],1)[:,::-1]
rec=np.c_[np.zeros(len(x)),x[:,1:]-pm[:,:-1]]
for i in [3,13,15,0,5]:
    y=h[:,i]
    print(i,'prefmax',np.round([np.corrcoef(y,pm[:,k])[0,1] for k in range(10)],2))
    print(i,'sufmax ',np.round([np.corrcoef(y,sm[:,k])[0,1] for k in range(10)],2))
    print(i,'relurec',np.round([np.corrcoef(y,np.maximum(rec[:,k],0))[0,1] for k in range(10)],2))

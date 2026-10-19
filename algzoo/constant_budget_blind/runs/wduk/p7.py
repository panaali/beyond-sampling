import numpy as np
x=np.load('x1.npy');h=np.load('h1.npy')
pm=np.maximum.accumulate(x,1)
sm=np.maximum.accumulate(x[:,::-1],1)[:,::-1]
for i in [1,2,5,8,9,11,13]:
    hi=h[:,i]
    print(i,'prefmax',' '.join('%.2f'%np.corrcoef(hi,pm[:,t])[0,1] for t in range(10)),'| sufmax',' '.join('%.2f'%np.corrcoef(hi,sm[:,t])[0,1] for t in range(10)))

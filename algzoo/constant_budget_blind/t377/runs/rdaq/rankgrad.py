import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
G=np.load('G2.npy');x=np.load('x2.npy');H=np.load('H2.npy')
srt=np.argsort(-x,1)
GR=np.take_along_axis(G,srt[:,None,:],2) # N,32,10 by rank
for i in range(32):
    a=H[:,i]>1e-4
    print(i,'rank-grad mean',GR[a,i,:].mean(0),' med',np.median(GR[a,i,:3],0))

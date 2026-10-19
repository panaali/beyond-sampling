import numpy as np
x=np.load('x2.npy');h=np.load('h2.npy')
p=x[:,:9]; s=np.sort(p,1)[:,::-1]; am=np.argmax(p,1)
M,S,T=s[:,0],s[:,1],s[:,2]
np.set_printoptions(precision=2,suppress=True,linewidth=220)
for i in [1,6,7,8,2,0,14]:
    print('neuron',i)
    for j in range(9):
        g=(am==j)&(h[:,i]>0.5)
        X=np.c_[np.ones(g.sum()),M[g],S[g],T[g],x[g,9]]
        c,*_=np.linalg.lstsq(X,h[g,i],rcond=None)
        r=h[g,i]-X@c; print('  am',j,'R2 %.3f'%(1-r.var()/h[g,i].var()),c)

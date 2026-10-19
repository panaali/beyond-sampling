import numpy as np,sys
x=np.load('x1.npy');h=np.load('h1.npy')
o=np.argsort(-x,1)
np.set_printoptions(precision=1,suppress=True,linewidth=150)
for i in map(int,sys.argv[1:]):
    print('neuron',i)
    for a in range(10):
        m=o[:,0]==a
        c,*_=np.linalg.lstsq(x[m],h[m,i],rcond=None)
        r=1-((h[m,i]-x[m]@c)**2).mean()/h[m,i].var()
        print(a,c,'r2 %.3f'%r, 'mean %.2f'%h[m,i].mean(), 'frac0 %.2f'%(h[m,i]==0).mean())

import numpy as np,sys
x=np.load('x1.npy');h=np.load('h1.npy')
rk=np.argsort(np.argsort(-x,1),1)
np.set_printoptions(precision=1,suppress=True,linewidth=150)
i=int(sys.argv[1]);j=int(sys.argv[2])
for r in range(10):
    m=rk[:,j]==r
    c,*_=np.linalg.lstsq(x[m],h[m,i],rcond=None)
    print('rank of x%d ='%j,r,c,'r2 %.2f mean %.1f f0 %.2f'%(1-((h[m,i]-x[m]@c)**2).mean()/h[m,i].var(),h[m,i].mean(),(h[m,i]==0).mean()))

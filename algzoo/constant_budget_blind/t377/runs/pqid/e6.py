import numpy as np
x=np.load('x2.npy');h=np.load('h2.npy')
N=len(x); o=np.argsort(-x,1); s=-np.sort(-x,1)
am,a2,a3=o[:,0],o[:,1],o[:,2]; M,S,T=s[:,0],s[:,1],s[:,2]
oh=lambda a:np.eye(10)[a]
np.set_printoptions(precision=2,suppress=True,linewidth=220)
def fit(X,y,name):
    g=y>0.3
    for it in range(4):
        c,*_=np.linalg.lstsq(X[g],y[g],rcond=None); g=(y>0.3)|(X@c>0)
    pred=np.maximum(X@c,0); r2=1-((y-pred)**2).mean()/y.var()
    print('  ',name,'R2 %.3f'%r2); return c
one=np.ones((N,1))
for i in [3,13,15,5,9,0,10]:
    print('neuron',i)
    c=fit(np.c_[one,x,M,S],h[:,i],'lin x+M+S'); print('    ',c)
    c=fit(np.c_[one,M[:,None]*oh(am),S[:,None]*oh(a2),T[:,None]*oh(a3)],h[:,i],'M*oh(am),S*oh(a2),T*oh(a3)'); print('    M',c[1:11]);print('    S',c[11:21]);print('    T',c[21:])
    c=fit(np.c_[one,M[:,None]*oh(am),x],h[:,i],'M*oh(am)+x'); print('    M',c[1:11]);print('    x',c[11:])

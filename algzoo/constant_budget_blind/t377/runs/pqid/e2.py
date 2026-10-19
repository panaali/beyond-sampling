import numpy as np, probe
rng=np.random.default_rng(2)
x=rng.standard_normal((200000,10))
h=probe.hidden(x)
np.save('x2.npy',x);np.save('h2.npy',h)
am=np.argmax(x,1); M=x.max(1)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
# per am group: regress h_i on [1, M, x0..x9] restricted to h_i>0
for i in range(16):
    print('neuron',i)
    for j in range(10):
        g=(am==j)&(h[:,i]>0.5)
        if g.sum()<200: print('  am',j,'n',g.sum()); continue
        X=np.c_[np.ones(g.sum()),M[g],x[g]]
        X[:,2+j]=0
        c,*_=np.linalg.lstsq(X,h[g,i],rcond=None)
        r=h[g,i]-X@c; r2=1-r.var()/h[g,i].var()
        print('  am',j,'n%6d'%g.sum(),'R2 %.3f'%r2,c)

import numpy as np, feats, probe
np.set_printoptions(precision=1,suppress=True,linewidth=230)
rng=np.random.default_rng(10)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
R=lambda z:np.maximum(z,0)
pre=x[:,:9]; ps=-np.sort(-pre,1)
D=np.stack([R(x[:,t]-np.delete(pre,t,1).max(1)) for t in range(9)],1)
E=np.stack([R(x[:,t]-(-np.sort(-np.delete(pre,t,1),1))[:,1]) for t in range(9)],1)
for name,Fm in [('D+ps',np.c_[D,ps[:,:3]]),('D+E+ps',np.c_[D,E,ps[:,:3]])]:
    print(name)
    for i in [0,1,2,4,5,6,10,11,12,14,15,21,23,24,26,27,29,31]:
        A=np.c_[Fm,np.ones(len(x))]; c=feats.relufit(H[:,i],A); p=R(A@c)
        print(' ',i,'R2 %.3f'%(1-((H[:,i]-p)**2).mean()/H[:,i].var()),c)

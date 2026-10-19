import numpy as np, probe, labels
x=np.random.default_rng(11).standard_normal((100000,10)); h=probe.hidden(x)
for i in range(16):
    p=labels.f(i,x); r2=1-((h[:,i]-p)**2).mean()/h[:,i].var()
    print(i,'R2 %.4f'%r2,'words',len(labels.LABELS[i][1].split()))
np.set_printoptions(precision=2,suppress=True)
for i,(b,c0,A,om,ph,sg) in labels._F.items(): print('w',i,c0+A*np.cos(om*np.arange(9)+ph))
for i,(u1,u2,c) in labels._Q.items(): print('Q',i,'gain1',c[0]+c[1]*np.array(u1),'gain2',c[2]+c[3]*np.array(u2))
for i,(u1,u2,c) in labels._R.items(): print('R',i,'w1',c[0]+c[1]*np.array(u1),'w2',c[2]+c[3]*np.array(u2))

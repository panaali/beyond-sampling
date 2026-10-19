import numpy as np, feats, probe
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(12)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
R=lambda z:np.maximum(z,0)
pre=x[:,:9]; q=-np.sort(-pre,1); am=pre.argmax(1); gap=q[:,0]-q[:,1]
def A_of(T,mode):
    G=np.isin(am,T)*gap
    if mode=='a': cols=[G,gap,q[:,0],q[:,1]]
    else: cols=[G,np.isin(am,T)*1.0,q[:,0],q[:,1]]
    return np.c_[np.stack(cols,1),np.ones(len(x))]
def score(i,T,mode):
    A=A_of(T,mode); c=feats.relufit(H[:,i],A); p=R(A@c); return 1-((H[:,i]-p)**2).mean()/H[:,i].var(),c
for i in [6,4,15,5,0,12,31,10,29]:
  for mode in 'ab':
    T=[]; best=-9; imp=True
    while imp:
        imp=False
        for u in range(9):
            T2=[v for v in T if v!=u] if u in T else sorted(T+[u])
            if not T2: continue
            r=score(i,T2,mode)[0]
            if r>best+1e-3: best=r;T=T2;imp=True
    r,c=score(i,T,mode)
    print(i,mode,'T',T,'R2 %.3f'%r,c)

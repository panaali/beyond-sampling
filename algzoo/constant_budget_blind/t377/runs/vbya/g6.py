import numpy as np, feats, probe
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(11)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
R=lambda z:np.maximum(z,0)
def featT(x,t,T,extra):
    pre=x[:,:t]; q=-np.sort(-pre,1); am=pre.argmax(1)
    D=np.isin(am,T)*(q[:,0]-q[:,1])
    cols=[x[:,t],q[:,0],q[:,1],D]
    if extra=='later': cols.append(x[:,t+1:].max(1) if t<9 else np.zeros(len(x)))
    return np.c_[np.stack(cols,1),np.ones(len(x))]
def score(i,t,T,extra):
    A=featT(x,t,T,extra); c=feats.relufit(H[:,i],A); p=R(A@c)
    return 1-((H[:,i]-p)**2).mean()/H[:,i].var(),c
for i,t in [(14,8),(31,8),(27,8),(10,8),(24,7),(26,7),(12,5),(23,6),(6,8)]:
  for extra in ['','later']:
    T=[]; best=score(i,t,T,extra)[0]; imp=True
    while imp:
        imp=False
        for u in range(t):
            T2=[v for v in T if v!=u] if u in T else sorted(T+[u])
            r=score(i,t,T2,extra)[0]
            if r>best+1e-3: best=r;T=T2;imp=True
    r,c=score(i,t,T,extra)
    print(i,'t',t,extra,'T',T,'R2 %.3f'%r,'coef(x_t,q1,q2,D,[later],1)',c)

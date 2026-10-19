import numpy as np, probe, sys
from dic4 import greedyR
from dic3 import fitrelu
r=lambda z:np.maximum(z,0)
def d7(x):
    s=np.sort(x,1)[:,::-1]; D={'s0':s[:,0],'s1':s[:,1],'g':s[:,0]-s[:,1]}
    am=np.argmax(x,1); E=(am%2==0)&(am<8); D['PE*s0']=E*s[:,0]; D['PO*s0']=((am%2==1)&(am<8))*s[:,0]
    M=np.maximum.accumulate(x,1)
    for a in range(10):
        D[f'x{a}']=x[:,a]; D[f'r(x{a})']=r(x[:,a]); D[f'r(-x{a})']=r(-x[:,a]); D[f'xn{a}']=x[:,a]*(am!=a)
        D[f'M{a}']=M[:,a]
        for b in range(10):
            if a<b and b<8:
                D[f'max(x{a},x{b})']=np.maximum(x[:,a],x[:,b]); D[f'min(x{a},x{b})']=np.minimum(x[:,a],x[:,b])
            if a!=b and a<8 and b<8: D[f'r(x{a}-x{b})']=r(x[:,a]-x[:,b])
        if a>0: D[f'cap{a}']=np.minimum(x[:,a],M[:,a-1]); D[f'r(x{a}-M{a-1})']=r(x[:,a]-M[:,a-1])
    return D
x=np.random.default_rng(3).standard_normal((6000,10)); h=probe.hidden(x); D=d7(x)
xt=np.random.default_rng(7).standard_normal((30000,10)); ht=probe.hidden(xt); Dt=d7(xt)
for i in [int(a) for a in sys.argv[1].split(',')]:
    g=greedyR(D,h[:,i],5,6); ks=[k for k,_ in g]
    b=fitrelu(np.stack([D[k] for k in ks],1),h[:,i])
    p=np.maximum(np.stack([Dt[k] for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
    print(i,' '.join(f'{k}({v:.3f})' for k,v in g),'| test %.3f'%r2, b.round(2),flush=True)

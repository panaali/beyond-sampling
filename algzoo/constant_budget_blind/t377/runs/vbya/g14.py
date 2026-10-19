import numpy as np, feats, probe, sys
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(19)
x=rng.standard_normal((12000,10)); H=probe.hidden(x)
xt=rng.standard_normal((20000,10)); Ht=probe.hidden(xt)
F,u9=feats.build3(x); Ft,_=feats.build3(xt); names=[n for n in F if not u9[n]]
R=lambda z:np.maximum(z,0)
print(len(names))
for i in [int(a) for a in sys.argv[1:]]:
    for fn in [feats.greedy,feats.greedy_act]:
        hist=fn(H[:,i],F,names,4)
        for r,sel,c in hist[1:]:
            At=np.c_[np.stack([Ft[s] for s in sel],1),np.ones(len(xt))]
            if fn is feats.greedy: p=At@c
            else: p=R(At@c)
            print(i,fn.__name__,'test %.3f'%(1-((Ht[:,i]-p)**2).mean()/Ht[:,i].var()),sel,c)

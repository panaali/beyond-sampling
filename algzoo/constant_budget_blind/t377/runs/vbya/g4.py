import numpy as np, feats, probe, sys
np.set_printoptions(precision=2,suppress=True,linewidth=220)
rng=np.random.default_rng(9)
x=rng.standard_normal((15000,10)); H=probe.hidden(x)
F,u9=feats.build2(x); names=list(F)
for i in [int(a) for a in sys.argv[1:]]:
    h1=feats.greedy(H[:,i],F,names,4)
    h2=feats.greedy_act(H[:,i],F,names,4)
    print(i,'LIN ',' | '.join('%.3f %s'%(r,s[-1]) for r,s,c in h1),h1[-1][2])
    print(i,'RELU',' | '.join('%.3f %s'%(r,s[-1]) for r,s,c in h2),h2[-1][2])

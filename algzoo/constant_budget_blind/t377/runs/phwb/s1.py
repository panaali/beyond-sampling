import numpy as np, probe
from feats import feats
x=np.random.default_rng(0).standard_normal((100000,10))
h=probe.hidden(x); F=feats(x)
np.save('/tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t377/phwb/x0.npy',x);np.save('h0.npy',h)
print('acc',(probe.predict(x)==F['a2']).mean())
for i in range(16):
    hi=h[:,i]
    line=f"{i:2d} mean{hi.mean():6.2f} sd{hi.std():5.2f} zero{(hi<1e-6).mean():4.2f} |"
    for k in range(10): line+=f"{np.corrcoef(hi,x[:,k])[0,1]:5.2f}"
    line+=" |"
    for k in ['m1','m2','m3']: line+=f"{np.corrcoef(hi,F[k])[0,1]:5.2f}"
    # mean by argmax pos and arg2 pos
    print(line)
for i in range(16):
    print(i,'byA1',np.round([h[F['a1']==p,i].mean() for p in range(10)],2),'byA2',np.round([h[F['a2']==p,i].mean() for p in range(10)],2))

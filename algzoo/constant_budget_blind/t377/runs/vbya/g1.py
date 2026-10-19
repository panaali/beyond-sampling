import numpy as np, feats
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy')[:20000]; H=np.load('H1.npy')[:20000]
F,u9=feats.build(x); names=list(F)
print(len(names))
import sys
for i in [int(a) for a in sys.argv[1:]]:
    hist=feats.greedy(H[:,i],F,names,4)
    print(i,' | '.join('%.3f %s'%(r,s[-1]) for r,s,c in hist), ' coef',hist[-1][2])

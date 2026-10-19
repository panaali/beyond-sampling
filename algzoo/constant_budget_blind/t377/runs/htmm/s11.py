import numpy as np,sys,itertools
from lib import fit_relu
from fe import base,feat
xa=np.load('x2.npy')[:12000]; ha=np.load('h2.npy')[:12000]
D=base(xa)
names=['x9','P8','Q8','R8','P7']+['x%d'%k for k in range(9)]+['d%d'%k for k in range(9)]+['q%d'%k for k in range(9)]
names+=['ABS0123456','ABS01234567']
for pre in ['AM','A2','T2']:
    for r in [1,2]:
        for S in itertools.combinations(range(9),r): names.append(pre+''.join(map(str,S)))
X=np.stack([feat(D,f) for f in names],1)
for i in [int(a) for a in sys.argv[1:]]:
    y=ha[:,i]; sel=[]; out=[]
    for step in range(4):
        best=(-9,None)
        for j in range(len(names)):
            if j in sel: continue
            b,r=fit_relu(X[:,sel+[j]],y,it=3)
            if r>best[0]: best=(r,j)
        sel.append(best[1]); out.append('%s(%.3f)'%(names[best[1]],best[0]))
    print(i,' '.join(out),flush=True)

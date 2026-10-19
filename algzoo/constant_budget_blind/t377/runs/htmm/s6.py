import numpy as np,sys
from lib import fit_relu
from lib2 import library2
x=np.load('x2.npy')[:15000]; h=np.load('h2.npy')[:15000]
F=library2(x); names=list(F); X=np.stack([F[k] for k in names],1)
neur=[int(a) for a in sys.argv[1:]]
for i in neur:
    y=h[:,i]; sel=[]; out=[]
    for step in range(4):
        best=(-9,None)
        for j in range(len(names)):
            if j in sel: continue
            b,r=fit_relu(X[:,sel+[j]],y,it=4)
            if r>best[0]: best=(r,j)
        sel.append(best[1]); out.append('%s(%.3f)'%(names[best[1]],best[0]))
    b,r=fit_relu(X[:,sel],y,it=12)
    print(i,' '.join(out),'| coefs',np.round(b,2),flush=True)

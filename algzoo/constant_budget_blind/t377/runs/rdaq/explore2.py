import numpy as np
from feats import *
x=np.load('x1.npy')[:40000];H=np.load('H1.npy')[:40000]
d=F(x)
groups={'x':['x'],'xP':['x','P'],'xPs':['x','P','s'],'all':['x','P','s','am','a2','rec']}
for i in range(32):
    y=H[:,i]; out=[]
    for g,ks in groups.items():
        out.append('%s=%.3f'%(g,r2(np.hstack([d[k] for k in ks]),y)[0]))
    print(i,' '.join(out))

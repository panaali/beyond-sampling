import numpy as np, probe
from forms import feats
from dic3 import relufit
C={0:[['M8','x8','x7','x9','dn4'],['M8','x8','x7','x9','x456']],
1:[['M8','S8','M8o','S8o','x6'],['M7','S7','M7o','S7o','x6']],
2:[['M8','S8','M8o','S8o','x9']],
3:[['M7','M7o','x4','x1','x2']],
4:[['M7','M7o','x6','Jodd7','x4']],
5:[['M8','x4','x8','dn7','x9']],
6:[['M7','S7','M7o','S7o','x7'],['M6','S6','M6o','S6o','x7']],
7:[['M8','M7','M6o','x5','x7'],['M8','M7','M7o','x5','x7']],
8:[['M8','up8','M7o','x8','dn5'],['M8','up8','M7o','x8','x5']],
9:[['M7','x5','x4','x2','x6'],['M7','x5','x4','x268']],
10:[['up9']],
11:[['M8','S8','M8o','S8o','x9']],
12:[['x3','dn1','up2','up3','dn4'],['x3','x1','min3','s1','x0'],['x3','x1','min3','M2' if False else 'S9','dn1']],
13:[['M8','up8','x2','dn4','up1'],['M8','up8','x2','x4','x1']],
14:[['g6x6','g6s1','g6x9']],
15:[['x5','x2','M7','M7o','up1'],['x5','x2','M7','M7o','x1']]}
xa=np.load('x1.npy'); ha=np.load('h1.npy'); Fa=feats(xa)
xb=np.random.default_rng(99).standard_normal((100000,10)); hb=probe.hidden(xb); Fb=feats(xb)
out={}
for i,cands in C.items():
    best=None
    for names in cands:
        c,_=relufit(np.stack([Fa[n] for n in names],1),ha[:,i])
        p=np.maximum(np.stack([Fb[n] for n in names],1)@c,0); r=1-((p-hb[:,i])**2).mean()/hb[:,i].var()
        print(i,names,'test R2 %.4f'%r,np.round(c,2))
        if best is None or r>best[0]: best=(r,names,c)
    out[i]=(best[1],[round(float(v),3) for v in best[2]],round(best[0],4))
import json; json.dump({str(k):v for k,v in out.items()},open('fit_out.json','w'))

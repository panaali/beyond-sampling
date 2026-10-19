import numpy as np, probe, json
from forms import feats
from dic3 import relufit
SEL={0:['M8','x8','x7','x9','x456'],1:['M8','S8','M8o','S8o','x6'],2:['M8','S8','M8o','S8o','x9'],
3:['M7','M7o','min4','x2','x1'],4:['M7','M7o','x6','Jodd7','x4'],5:['M8','x4','x8','dn7','x9'],
6:['M7','S7','M7o','S7o','x7'],7:['M8','M7','M6o','x5','x7'],8:['M8','up8','M7o','x8','dn5'],
9:['M7','x456','x2','min5','M8o'],10:['up9'],11:['M8','S8','M8o','S8o','x9'],
12:['x3','dn1','up2','up3','dn4'],13:['up8','M7','x456','x2','up1'],14:['g6x6','g6s1','g6x9'],
15:['min5','x2','M7','M7o','up1']}
xa=np.r_[np.load('x1.npy'),np.random.default_rng(3).standard_normal((100000,10))]
ha=np.r_[np.load('h1.npy'),probe.hidden(xa[100000:])]
Fa=feats(xa); out={}
for i,n in SEL.items():
    c,r=relufit(np.stack([Fa[k] for k in n],1),ha[:,i],40); out[i]=(n,[round(float(v),3) for v in c]); print(i,n,np.round(c,2),'%.4f'%r)
json.dump({str(k):v for k,v in out.items()},open('sel.json','w'))

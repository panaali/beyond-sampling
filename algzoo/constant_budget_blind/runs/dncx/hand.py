import numpy as np, json, sys
from featlib import base, feat
from lib2 import fitrelu
d=np.load('d1.npz'); x=d['x'][:40000].astype(float); h=d['h'][:40000]
e=np.load('d2.npz'); xe=e['x'].astype(float); he=e['h']
b=base(x); be=base(xe)
S=json.load(open(sys.argv[1]))
res=json.load(open('fits.json')) if len(sys.argv)>2 else {}
for k,specs in S.items():
    i=int(k.split('_')[0]); X=np.c_[[feat(b,s) for s in specs]].T; r2,c=fitrelu(X,h[:,i],15)
    p=np.maximum(np.c_[[feat(be,s) for s in specs]].T@c,0); r2e=1-((he[:,i]-p)**2).mean()/he[:,i].var()
    print(i,'train %.3f test %.3f'%(r2,r2e),list(zip(specs,np.round(c,3))))
    res[k]=dict(specs=specs,coefs=[round(float(q),4) for q in c],r2=round(float(r2e),4))
json.dump(res,open('fits.json','w'),indent=0)

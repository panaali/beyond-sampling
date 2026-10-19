import numpy as np, json
from lib import feats
x=np.load('x1.npy')[:8000];H=np.load('H1.npy')[:8000]; F=feats(x)
xv=np.load('x1.npy')[80000:100000];Hv=np.load('H1.npy')[80000:100000]; Fv=feats(xv)
cols=['A_P6','am8','r(x0-to)','r(x1-to)']; y=H[:,28]
A=np.stack([F[c] for c in cols]+[np.ones(len(y))],1); m=np.ones(len(y),bool)
for _ in range(12):
    c,*_=np.linalg.lstsq(A[m],y[m],rcond=None); m=(y>1e-6)|(A@c>0)
Av=np.stack([Fv[cc] for cc in cols]+[np.ones(len(xv))],1); yv=Hv[:,28]
r=1-((np.maximum(Av@c,0)-yv)**2).mean()/yv.var(); print(r,c)
d=json.load(open('fit_16.json')); d['28']=dict(cols=cols,c=[float(v) for v in c],r2=float(r)); json.dump(d,open('fit_16.json','w'))

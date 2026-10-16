import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
fc=faces23(512)
rho=np.linalg.norm(fc['x0'],axis=1); x0=fc['x0']/rho[:,None]; d=fc['d']
r=np.full(512,1e-12); recs=[]
for c in range(6):
    L,dL,P,dP=fwd_jvp(M,x0+r[:,None]*d,d)
    with np.errstate(divide='ignore',invalid='ignore'): root=-P/dP
    root[~(root>0)]=np.inf; rf=root.reshape(512,-1); j=rf.argmin(1)
    recs.append((r.copy(), j//16, j%16)); r=r+rf.min(1)+1e-12
import collections
for c,(rr,t,i) in enumerate(recs):
    print('cell',c,'start median',np.median(rr), 'q10',np.quantile(rr,.1), collections.Counter(zip(t.tolist(),i.tolist())).most_common(6))
# how does flip time relate to positions?
a=fc['tpos'][:512]; b=fc['tneg'][:512]
t0=recs[1][1]; print('flip time vs max(a,b):',collections.Counter((t0-np.maximum(a,b)).tolist()).most_common(8))

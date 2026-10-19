import numpy as np, sys, json
from util import *; from dict2 import base; from featlib import design
x=np.load('x200.npy')[:20000]; H=np.load('H200.npy')[:20000]
D,o,s=base(x)
names=list(D); cols=[D[n] for n in names]
for S in range(1,1024):
    idx=[j for j in range(10) if S>>j&1]
    if len(idx)<2: continue
    srt=np.sort(x[:,idx],1)
    names.append('max{'+','.join(map(str,idx))+'}'); cols.append(srt[:,-1])
    names.append('sec{'+','.join(map(str,idx))+'}'); cols.append(srt[:,-2])
g8=(x[:,:9].argmax(1)!=8)
A=np.stack(cols,1); Ag=A*g8[:,None]
C=json.load(open('c2.json'))
def quick(F,h,it):
    w=np.linalg.lstsq(F,h,rcond=None)[0]
    for _ in range(it):
        m=(h>0)|(F@w>0); w=np.linalg.lstsq(F[m],h[m],rcond=None)[0]
    return w, r2(np.maximum(F@w,0),h)
for i in [int(a) for a in sys.argv[1:]]:
    h=H[:,i]; F0=design(x,C[str(i)]); res=[]
    for k in range(len(names)):
        for gated,AA in (('',A),('g',Ag)):
            w,r=quick(np.hstack([F0,AA[:,k:k+1]]),h,4); res.append((r,names[k]+gated))
    res.sort(reverse=True); print(i,res[:5],flush=True)

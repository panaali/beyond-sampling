import numpy as np, json
from lib import base
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:10000].astype(float); h=d["h"][:10000].astype(float)
F=base(x); G=json.load(open("s10.json"))
out={}
for i in [0,1,2,4,5,6,10,11,12,15,21,23,24,26,29,31,19,3,20,18,22,28,30,17,13,14,9]:
    y=h[:,i]; nm=list(F); M=[F[n] for n in nm]
    if str(i) in G:
        g=G[str(i)][0]
        for gg in (1,2,3):
            idx=[j for j in range(10) if g[j]==gg]
            if idx and len(idx)>1: nm.append("gmax%s"%"".join(map(str,idx))); M.append(x[:,idx].max(1))
    M=np.stack(M,1)
    sel=[];rs=[]
    for step in range(5):
        best=(-9,None,None)
        for k in range(M.shape[1]):
            if k in sel: continue
            c,r=relufit(M[:,sel+[k]],y,iters=10)
            if r>best[0]: best=(r,k,c)
        sel.append(best[1]); rs.append(round(best[0],3))
    out[i]=([nm[k] for k in sel],list(best[2]),rs)
    print(i,rs,[nm[k] for k in sel],np.round(best[2],2),flush=True)
json.dump(out,open("s11.json","w"))

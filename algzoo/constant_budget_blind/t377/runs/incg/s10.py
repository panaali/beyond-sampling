import numpy as np, json
from rfit import relufit, r2
from feats2 import P5
d=np.load("d1.npz"); x=d["x"][:8000].astype(float); h=d["h"][:8000].astype(float)
S=-np.sort(-x,1); S1=S[:,1]; S2=S[:,2]
LOW=-4.0
def gm(g,G):
    idx=[j for j in range(10) if G[j]==g]
    return x[:,idx].max(1) if idx else None
def design(G):
    cols=[]
    for g in (1,2,3):
        m=gm(g,G)
        if m is not None: cols.append(m)
    cols+= [S1,S2]
    return np.stack(cols,1)
def score(y,G):
    c,r=relufit(design(G),y,iters=10); return r,c
res={}
for i in range(32):
    y=h[:,i]
    if y.var()<1e-3: print(i,"dead"); continue
    cors=[np.corrcoef(y,x[:,j])[0,1] for j in range(10)]
    k=int(np.argmax(np.abs(cors)))
    inits=[]
    G=[2]*10; G[k]=1; inits.append(G)
    am=np.argmax(x,1); mm=np.array([y[am==m].mean() for m in range(10)])
    G=[1 if mm[j]>np.median(mm) else 2 for j in range(10)]; inits.append(G)
    if P5.get(i):
        G=[1 if j in P5[i] else 2 for j in range(10)]; G[9]=3 if i not in (14,24,12,23,13) else G[9]; inits.append(G)
    G=[1 if mm[j]>np.median(mm) else 2 for j in range(10)]; G[k]=3; inits.append(G)
    best=(-9,None,None)
    for G in inits:
        r,c=score(y,G)
        improved=True
        while improved:
            improved=False
            for j in range(10):
                for g in (0,1,2,3):
                    if g==G[j]: continue
                    G2=list(G); G2[j]=g; r2_,c2=score(y,G2)
                    if r2_>r+2e-4: G,r,c=G2,r2_,c2; improved=True
        if r>best[0]: best=(r,G,c)
    res[i]=(best[1],list(best[2]))
    print(i,"R2 %.3f"%best[0],"groups",best[1],np.round(best[2],2),flush=True)
json.dump(res,open("s10.json","w"))

import numpy as np, json, probe
from lib import base
from rfit import relufit, r2
G=json.load(open("s10.json")); Gr=json.load(open("s11.json"))
def featget(F,x,n):
    if n in F: return F[n]
    if n.startswith("gmax"): return x[:,[int(c) for c in n[4:]]].max(1)
    raise KeyError(n)
def gspec(i):
    g=G[str(i)][0]; names=[]
    for gg in (1,2,3):
        idx=[j for j in range(10) if g[j]==gg]
        if idx: names.append("gmax"+"".join(map(str,idx)) if len(idx)>1 else "x%d"%idx[0])
    return names+["S1","S2"]
rng=np.random.default_rng(11); xa=rng.standard_normal((30000,10)); ha=np.asarray(probe.hidden(xa.astype(np.float32))).astype(float)
rng=np.random.default_rng(12); xb=rng.standard_normal((50000,10)); hb=np.asarray(probe.hidden(xb.astype(np.float32))).astype(float)
Fa=base(xa); Fb=base(xb)
cands={}
for i in range(32):
    c=[]
    if str(i) in G: c.append(gspec(i))
    if str(i) in Gr:
        nm=Gr[str(i)][0]; c+= [nm, nm[:4], nm[:3], nm[:2]]
    cands[i]=c
cands[16]=[["inTop2_7","rec7","mex7","below7"]]; cands[25]=[["x9","sm1","p2nd2","pm8"],["S0","x9"]]
chosen={}
for i in range(32):
    rows=[]
    for nm in cands[i]:
        A=np.stack([featget(Fa,xa,n) for n in nm],1); c,_=relufit(A,ha[:,i],iters=30)
        B=np.stack([featget(Fb,xb,n) for n in nm],1); r=r2(hb[:,i],np.maximum(B@c,0))
        rows.append((r,len(nm),nm,list(c)))
    best=max(r for r,*_ in rows)
    ok=[t for t in rows if t[0]>=best-0.01]; ok.sort(key=lambda t:(t[1],-t[0]))
    chosen[i]=ok[0]
    print(i," | ".join("%.3f/%d"%(r,n) for r,n,_,_ in rows)," -> %.3f"%ok[0][0],ok[0][2],np.round(ok[0][3],2),flush=True)
json.dump({i:[v[2],v[3],v[0]] for i,v in chosen.items()},open("chosen.json","w"))

import numpy as np, probe, sys, json
from dic5 import dic5
from dic3 import fitrelu
x=np.random.default_rng(11).standard_normal((5000,10)); h=probe.hidden(x); D=dic5(x)
xt=np.random.default_rng(7).standard_normal((40000,10)); ht=probe.hidden(xt); Dt=dic5(xt)
keys=list(D); X=np.stack([D[k] for k in keys],1); X=X/(X.std(0)+1e-9)
neur=[int(a) for a in sys.argv[1].split(',')]; out={}
def err(cols,y):
    A=X[:,list(cols)]; b=np.linalg.lstsq(A,y,rcond=None)[0]
    for _ in range(4):
        m=(A@b>0)|(y>0); b=np.linalg.lstsq(A[m],y[m],rcond=None)[0]
    return ((y-np.maximum(A@b,0))**2).mean()
for i in neur:
    y=h[:,i]; beam=[()]
    for step in range(5):
        cand={}
        for B in beam:
            for j in range(len(keys)):
                if j in B: continue
                c=tuple(sorted(B+(j,)))
                if c not in cand: cand[c]=err(c,y)
        beam=[c for c,_ in sorted(cand.items(),key=lambda t:t[1])[:12]]
    res=[]
    for c in beam[:5]:
        ks=[keys[j] for j in c]; b=fitrelu(np.stack([D[k] for k in ks],1),y)
        p=np.maximum(np.stack([Dt[k] for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
        res.append((r2,ks,b.tolist()))
    res.sort(key=lambda t:-t[0])
    for r2,ks,b in res[:3]: print(i,'test %.3f'%r2,ks,np.round(b,2),flush=True)
    out[i]=res
json.dump(out,open(f'p13_{sys.argv[2]}.json','w'))

import numpy as np, probe, json
from dic3 import fitrelu
r=lambda z:np.maximum(z,0)
def base(x,kind):
    s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1); po=(am%2==1)&(am<8)
    if kind=='s0': return [s[:,0], po*s[:,0], (am>=8)*s[:,0]]
    if kind=='s0b': return [s[:,0], po*s[:,0], (am==8)*s[:,0]]
    if kind=='M8':
        M8=x[:,:9].max(1); a8=np.argmax(x[:,:9],1); return [M8,(a8%2==1)*M8,(a8==8)*M8]
def xn(x): am=np.argmax(x,1); return [x[:,t]*(am!=t) for t in range(10)]
x=np.random.default_rng(13).standard_normal((20000,10)); h=probe.hidden(x)
xt=np.random.default_rng(7).standard_normal((40000,10)); ht=probe.hidden(xt)
out={}
for i in range(16):
    best=None
    for kind in ['s0','s0b','M8']:
        B=base(x,kind); XN=xn(x); y=h[:,i]
        bf=fitrelu(np.stack(B+XN,1),y); w=bf[3:]; order=np.argsort(w)
        for a in range(0,11):
            for c in range(a,11):
                segs=[order[:a],order[a:c],order[c:]]
                for z in range(3):
                    S=[sg for k,sg in enumerate(segs) if k!=z]
                    if any(len(sg)==0 for sg in S): continue
                    F=B+[sum(XN[t] for t in sg) for sg in S]
                    b=fitrelu(np.stack(F,1),y,8); e=((y-r(np.stack(F,1)@b))**2).mean()
                    if best is None or e<best[0]: best=(e,kind,[sorted(map(int,sg)) for sg in S],b)
    e,kind,S,b=best
    Bt=base(xt,kind); XNt=xn(xt); F=Bt+[sum(XNt[t] for t in sg) for sg in S]
    p=r(np.stack(F,1)@b); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
    print(i,kind,S,'test %.3f'%r2,b.round(2),flush=True); out[i]=(kind,S,b.tolist(),r2)
json.dump(out,open('p15.json','w'))

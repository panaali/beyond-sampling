import sys,json,numpy as np
sys.path.insert(0,"m16")
from arc_eval import forward, noisy_models, Wi0,Wh0,Wo0
from stratified import stratum, N_STRATA
d=json.load(open("m16/stratified_design.json")); P=np.array(d["P"]); al=np.array(d["alloc"])
rng=np.random.default_rng(7); x=rng.standard_normal((400_000,10)); h=stratum(x); tg=np.argsort(x,1)[:,-2]
def gain(m,B=16000):
    ok=forward(m,x).argmax(1)==tg
    p=np.array([ok[h==k].mean() for k in range(N_STRATA)])
    acc=(P*p).sum()
    n=np.maximum(2,np.round(al*B))
    vs=(P**2*p*(1-p)/n).sum(); vm=acc*(1-acc)/n.sum()
    neyman=P*np.sqrt(p*(1-p)); neyman/=neyman.sum(); vn=(P**2*p*(1-p)/np.maximum(2,np.round(neyman*B))).sum()
    return acc, vm/vs, vm/vn, p
a,g,gn,p=gain((Wi0,Wh0,Wo0)); print("orig acc %.4f gain %.2f oracle-neyman %.2f"%(a,g,gn), np.round(p,3))
for s in (0.0025,0.005):
    gs=[gain(m)[1:3] for m in noisy_models(12,s,0)]
    print(s,"design gain per copy",np.round([q[0] for q in gs],2),"mean %.2f"%np.mean([q[0] for q in gs]))
# expected rejection cost per eval
cnt=np.bincount(h,minlength=N_STRATA)/len(h); print("candidates per eval:", (al/cnt).sum())

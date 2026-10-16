import sys, os, json, time, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,"../.."))
from arc_eval import forward, noisy_models, true_acc, est_sampling, Wi0, Wh0, Wo0
import stratified as S, cv
level=sys.argv[1]; reps=int(sys.argv[2]); budgets=[int(b) for b in sys.argv[3].split(",")]
d=json.load(open(os.path.join(HERE,"../../stratified_design.json"))); SP,SA=np.array(d["P"]),np.array(d["alloc"])
al=json.load(open(os.path.join(HERE,"alloc.json")))
ests={"sampling":est_sampling,
      "stratified":lambda m,b,rng:S.estimate(lambda x:forward(m,x),b,rng,SP,SA),
      "rb":cv.make(),
      "rb_cvB":cv.make(cv="B"),
      "srb":cv.make(strat=True,alloc=np.array(al["plain"])),
      "srb_cvA":cv.make(strat=True,cv="A",alloc=np.array(al["cv"]))}
if level=="0": models=[(Wi0,Wh0,Wo0)]; truth=[0.95317945]
else:
    models=noisy_models(6,float(level)); cache=os.path.join(HERE,f"truth_{level}.json")
    if os.path.exists(cache): truth=json.load(open(cache))
    else: truth=[float(true_acc(m,n=4_000_000)) for m in models]; json.dump(truth,open(cache,"w"))
res={"level":level,"truth":truth,"rows":[]}
for b in budgets:
    for name,e in ests.items():
        t=time.time(); se=[];used=[];errs=[]
        for mi,m in enumerate(models):
            for r in range(reps):
                v,u=e(m,b,np.random.default_rng(10_000*mi+r+7*b)); se.append((v-truth[mi])**2); used.append(u); errs.append(v-truth[mi])
        row=dict(budget=b,est=name,mse=float(np.mean(se)),evals=float(np.mean(used)),bias=float(np.mean(errs)),secs=time.time()-t)
        res["rows"].append(row); print(json.dumps(row),flush=True)
json.dump(res,open(os.path.join(HERE,f"results_{level}.json"),"w"),indent=1)

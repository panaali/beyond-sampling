import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import json
from pathlib import Path
import time
import numpy as np
from tail import conditional_probability,prefix_summary


def independent_validation(model,prefix,n,seed):
    """Direct network execution under the exact conditional radius law."""
    rng=np.random.default_rng(seed)
    direction=np.asarray(prefix)/np.linalg.norm(prefix)
    k=len(prefix)
    radius=np.sqrt(rng.chisquare(k,size=n))
    x=np.column_stack((direction[None,:]*radius[:,None],rng.standard_normal((n,2))))
    wi,wh,wo=model
    h=np.zeros((n,len(wi)))
    for t in range(x.shape[1]):
        h=np.maximum(h@wh.T+x[:,t,None]*wi,0)
    predicted=np.argmax(h@wo.T,axis=1)
    target=np.argsort(x,axis=1)[:,-2]
    indicators=predicted==target
    p=float(indicators.mean())
    return p,float(np.sqrt(p*(1-p)/n))


def main():
    cpu=time.process_time()
    cases=[]
    for sign in (-1.,1.):
        for label,wo in (("constant",np.zeros((3,1))),
                         ("last",np.array([[0.],[0.],[1.]]))):
            cases.append((f"H1T3_{label}_{sign}",
                          (np.array([1.]),np.zeros((1,1)),wo),np.array([sign])))
    for T in (3,5,10):
        wi=np.array([.8,-.6]);wh=np.array([[.35,-.7],[.2,.45]])
        wo=np.column_stack((np.cos(np.arange(T)*1.7),np.sin(np.arange(T)*1.7)))
        prefix=np.array([(-1)**i*(i+1)/(T-1) for i in range(T-2)])
        cases.append((f"H2T{T}",(wi,wh,wo),prefix))
    rows=[]
    for case,(name,model,prefix) in enumerate(cases):
        h,top=prefix_summary(model,prefix)
        try:
            p,stats=conditional_probability(model,h,top,len(prefix))
            scaled=tuple(w.copy() for w in model)
            scaled=(scaled[0]*1e-16,scaled[1],scaled[2]*1e-16)
            sh,st=prefix_summary(scaled,prefix)
            sp,ss=conditional_probability(scaled,sh,st,len(prefix))
            mc,se=independent_validation(model,prefix,65536,55000+case)
            rows.append(dict(case=name,probability=p,mc=mc,se=se,
                             z=(p-mc)/se if se else None,
                             scale_difference=sp-p,stats=stats,scale_stats=ss,
                             validation_draws=65536))
        except ArithmeticError as error:
            rows.append(dict(case=name,error=str(error)))
    output=dict(cases=rows,cpu_seconds=time.process_time()-cpu,
                validation_model_inputs=sum(r.get("validation_draws",0) for r in rows),
                primitive_model_inputs=0,
                deterministic_prefix_sequences=2*len(cases),
                deterministic_prefix_steps=2*sum(len(prefix) for _,_,prefix in cases),
                complete_estimator=False,
                failures=[r["case"] for r in rows if "error" in r])
    Path(__file__).with_name("tail_results.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output))


if __name__=="__main__":
    main()

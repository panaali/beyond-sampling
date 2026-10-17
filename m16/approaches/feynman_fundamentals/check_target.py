import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import json,time
from pathlib import Path
import numpy as np
from target_tail import target_probability
from tail import conditional_probability,prefix_summary


def direct(model,a,b,count,j,n,seed):
    rng=np.random.default_rng(seed)
    wi,wh,wo=model;T=len(wo);k=T-2
    radius=np.sqrt(rng.chisquare(k,n))
    target=radius*b
    fresh=rng.normal(size=(n,2))
    hidden=radius[:,None]*a
    if j<=T-3:
        values=[fresh[:,0],fresh[:,1]]
    elif j==T-2:
        values=[fresh[:,0],target,fresh[:,1]]
    else:
        values=[fresh[:,0],fresh[:,1],target]
    for x in values:
        hidden=np.maximum(hidden@wh.T+wi*x[:,None],0)
    prediction=np.argmax(hidden@wo.T,axis=1)
    rank_count=count+np.sum(fresh>target[:,None],axis=1)
    hit=(prediction==j)&(rank_count==1)
    p=float(hit.mean());se=float(np.sqrt(p*(1-p)/n))
    return p,se


start=time.process_time()
wi=np.array([.8,-.6]);wh=np.array([[.35,-.7],[.2,.45]])
def model(T,tied=False):
    wo=np.ones((T,2)) if tied else np.column_stack((np.cos(np.arange(T)*1.7),np.sin(np.arange(T)*1.7)))
    return wi,wh,wo
cases=[]
for j in range(3):
    for b in (-1.,1.):
        a=np.maximum(wi*b,0) if j==0 else np.zeros(2)
        cases.append((f"T3_j{j}_b{b}",model(3),a,b,0,j))
cases.extend([
    ("zero_count0",model(5),np.zeros(2),0.,0,0),
    ("zero_count1",model(5),np.zeros(2),0.,1,0),
    ("axis_penultimate",model(5),np.array([.7,0]),.2,1,3),
    ("tied_target0",model(5,True),np.zeros(2),.3,0,0),
    ("tied_target1",model(5,True),np.zeros(2),.3,0,1),
    ("absorbed_count2",model(5),np.zeros(2),0.,2,4),
])
rows=[]
for i,(name,weights,a,b,count,j) in enumerate(cases):
    p,stats=target_probability(weights,a,b,count,j)
    scaled=(weights[0]*1e-16,weights[1],weights[2]*1e-16)
    q,scaled_stats=target_probability(scaled,a*1e-16,b,count,j)
    mc,se=direct(weights,a,b,count,j,65536,55400+i)
    rows.append(dict(case=name,probability=p,mc=mc,se=se,z=(p-mc)/se if se else None,
                     scale_difference=q-p,stats=stats,scaled_stats=scaled_stats))
target_sum=sum(r["probability"] for r in rows[:6])/2
old_sum=0.;old_stats=[]
for b in (-1.,1.):
    a,top=prefix_summary(model(3),np.array([b]))
    p,stats=conditional_probability(model(3),a,top,1)
    old_sum+=p/2;old_stats.append(stats)
output=dict(cases=rows,T3_target_decomposition=target_sum,T3_top_two_decomposition=old_sum,
            decomposition_difference=target_sum-old_sum,
            cpu_seconds=time.process_time()-start,validation_draws=12*65536,
            validation_chunk=65536,extra_top_two_stats=old_stats,
            complete_estimator=False)
Path(__file__).with_name("target_results.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({k:v for k,v in output.items() if k not in ("cases","extra_top_two_stats")}))
print(json.dumps(dict(max_abs_z=max(abs(r["z"]) for r in rows if r["z"] is not None),
                     max_scale_difference=max(abs(r["scale_difference"]) for r in rows),
                     positive_fallbacks=sum(r["stats"]["positive_fallbacks"]+r["scaled_stats"]["positive_fallbacks"] for r in rows))))

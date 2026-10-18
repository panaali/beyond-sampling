"""Original complete controls plus exact/neighbor tolerance threshold probes."""
from pathlib import Path
import json,time
import numpy as np
from radial_probe import cases,Grouped,Audit
from selector import Guarded,scalar_bound,scalar_radial,ORDERS
start=time.process_time();v=cases();a=Audit();b=Audit();old=Grouped(a);new=Guarded(b)
t=time.process_time();ov,om=old.many(v,6,1e-11,128);oldcpu=time.process_time()-t
t=time.process_time();nv,nm=new.many(v,6,1e-11,128);newcpu=time.process_time()-t
different=[i for i,(x,y) in enumerate(zip(om,nm)) if x!=y]
assert not different and np.array_equal(ov,nv)
thresholds=[]
for order in sorted(set(m['quadrature_order'] for m in om)):
    index=next(i for i,m in enumerate(om) if m['quadrature_order']==order)
    lo,hi,mid,ratio=scalar_radial(v[index]);bound=scalar_bound(mid,ratio,order)
    for tolerance in (np.nextafter(bound,0.),bound,np.nextafter(bound,np.inf)):
        oldmeta=None;oldfailure=None
        try:oldmeta=Grouped(Audit()).select(v[index],6,float(tolerance),128)
        except ArithmeticError as exc:oldfailure=type(exc).__name__
        guarded=Guarded(Audit());newmeta=None;newfailure=None
        try:newmeta=guarded.select_many(v[index:index+1],6,float(tolerance),128)[0]
        except ArithmeticError as exc:newfailure=type(exc).__name__
        assert oldmeta==newmeta and oldfailure==newfailure
        thresholds.append(dict(index=index,reference_order=order,tolerance=float(tolerance),
                               order=None if newmeta is None else newmeta['quadrature_order'],
                               failure=newfailure,scalar_rescues=guarded.scalar_rescues))
empty=Guarded(Audit());ev,em=empty.many(np.empty((0,3,6)),6,1e-11,128);assert len(ev)==len(em)==0
try:Guarded(Audit(),max_nodes=1).many(v[:1],6,1e-11,128)
except ValueError:node_cap=True
else:raise AssertionError('node cap not enforced')
out=dict(cases=len(v),orders=sorted(set(m['quadrature_order'] for m in om)),metadata_differences=different,
         exact_values_equal=np.array_equal(ov,nv),scalar_rescues=new.scalar_rescues,
         scalar_checks=new.scalar_checks,vector_trials=new.vector_trials,
         old_cpu=oldcpu,new_cpu=newcpu,old_selection_cpu=old.selection_cpu,new_selection_cpu=new.selection_cpu,
         old_cost=a.report(),new_cost=b.report(),thresholds=thresholds,empty_pass=True,node_cap_pass=node_cap,
         cpu_seconds=time.process_time()-start)
(Path(__file__).resolve().parent/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:x for k,x in out.items() if k not in ('thresholds','old_cost','new_cost')}))

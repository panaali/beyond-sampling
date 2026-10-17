import json,time,resource,sys
from pathlib import Path
import numpy as np
from outer import grouped_mass
BASE=Path(__file__).parent;PARENT=BASE.parent
sys.path.insert(0,str(PARENT))
from models import load,noisy
from geometry import Geometry
from volume import plan
from segment import grouped_mass as segments
sys.path.pop(0)
start=time.process_time();out={'models':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'full.json').write_text(json.dumps(out,indent=2)+'\n')
t=time.process_time();base,sha=load(5);out['shared_checkpoint_loading_cpu_seconds']=time.process_time()-t
references={0.:.2792373713559676,.0025:.31063601436566657,.02:.32506108244184867}
for sigma in (0.,.0025,.02):
 s=time.process_time();model=base if sigma==0 else noisy(base,sigma);g=Geometry(25);row=dict(sigma=sigma,checkpoint_sha256=sha)
 try:
  cells=g.gates(model);correct=g.correct(model,cells);rays=g.simplices(correct);row['geometry_cpu_seconds']=time.process_time()-s;row['geometry_counts']=dict(g.counts)
  t=time.process_time();orders,bounds=plan(rays,1e-4,jacobi=True,support=True);row['planning_cpu_seconds']=time.process_time()-t
  row['tetra']=grouped_mass(rays,orders,bounds,deadline=start+105,inner_budget=1e-7);row['cold_cpu_excluding_shared_loading']=time.process_time()-s
  row['cold_cpu_charging_shared_loading']=row['cold_cpu_excluding_shared_loading']+out['shared_checkpoint_loading_cpu_seconds']
  row['previous_reference']=references[sigma];row['reference_difference']=abs(row['tetra']['estimate']-references[sigma])
  row['same_rule_I5_comparison']=segments(rays,orders,bounds,deadline=start+105,jacobi=True)
  row['same_rule_difference']=abs(row['tetra']['estimate']-row['same_rule_I5_comparison']['estimate'])
  if sigma==0:
   partition=g.simplices([v for v,a in cells]);ns,bs=plan(partition,1e-4,jacobi=True,support=True);row['partition']=grouped_mass(partition,ns,bs,deadline=start+105,inner_budget=1e-7);row['partition_error']=abs(row['partition']['estimate']-1)
 except Exception as exc:row['failure']=repr(exc)
 out['models'].append(row);save();print(row,flush=True)
 if time.process_time()-start>105:break
save()

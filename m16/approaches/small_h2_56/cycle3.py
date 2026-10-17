import json,time,resource
from pathlib import Path
import numpy as np
from geometry import Geometry
from volume import plan,grouped_mass,bernstein_bound
from models import load,noisy,sample
BASE=Path(__file__).parent;start=time.process_time();out={'models':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle3.json').write_text(json.dumps(out,indent=2)+'\n')
for t,sigma in ((5,.0025),(5,.02),(6,0.)):
 model,sha=load(t);model=noisy(model,sigma);row=dict(T=t,sigma=sigma,sha256=sha,integrals=[]);out['models'].append(row);g=Geometry(20)
 try:
  cells=g.gates(model);correct=g.correct(model,cells);rays=g.simplices(correct);allrays=g.simplices([v for v,a in cells]);row['geometry_counts']=g.counts.copy()
  row['correct_simplices']=len(rays);row['partition_simplices']=len(allrays);save();print('geometry',t,sigma,row['geometry_counts'],flush=True)
  if t==6:np.savez_compressed(BASE/'t6_cones.npz',rays=rays,partition=allrays)
  for name,vs in (('accuracy',rays),('partition',allrays)):
   if t==5:orders,bounds=plan(vs,1e-4)
   else:orders=np.full(len(vs),4);bounds=np.array([bernstein_bound(v,4) for v in vs])
   nodes=sum(int(n)**(t-1) for n in orders)
   if nodes>300_000_000:row['integrals'].append(dict(name=name,skipped='planned nodes exceed cycle guard',planned_nodes=nodes));save();continue
   value=grouped_mass(vs,orders,bounds,deadline=start+115);value['name']=name;row['integrals'].append(value);save();print('integral',value,flush=True)
 except (ArithmeticError,TimeoutError) as e:row['error']=str(e);row['geometry_counts']=g.counts.copy()
 if t==5:row['validation']=sample(model)
 save()
save()

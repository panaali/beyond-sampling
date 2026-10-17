import json,time,resource
from pathlib import Path
import numpy as np
from geometry import Geometry
from volume import plan,grouped_mass,bernstein_bound
from models import load,noisy
BASE=Path(__file__).parent;start=time.process_time();out={'models':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle4.json').write_text(json.dumps(out,indent=2)+'\n')
def hidden(model,x):
 wi,wh,wo=model;h=np.zeros(len(wi))
 for a in x:h=np.maximum(wh@h+wi*a,0)
 return h
for t,sigma in ((5,.0025),(5,.02),(6,0.)):
 model,sha=load(t);model=noisy(model,sigma);row=dict(T=t,sigma=sigma,sha256=sha,integrals=[]);out['models'].append(row);g=Geometry(25);setup=time.process_time()
 try:
  cells=g.gates(model);row['gate_affine_validation_max_error']=max(float(np.max(np.abs(hidden(model,v.mean(axis=0))-a@v.mean(axis=0)))) for v,a in cells)
  correct=g.correct(model,cells);rays=g.simplices(correct);row['geometry_counts']=g.counts.copy();row['setup_cpu_seconds']=time.process_time()-setup
  label=f't{t}_sigma{sigma}';np.savez_compressed(BASE/(label+'_cones.npz'),rays=rays)
  row['correct_simplices']=len(rays);save();print('geometry',t,sigma,row['geometry_counts'],flush=True)
  if t==5:
   orders,bounds=plan(rays,1e-4);value=grouped_mass(rays,orders,bounds,deadline=start+115);value['name']='accuracy';row['integrals'].append(value);save();print(value,flush=True)
  else:
   allrays=g.simplices([v for v,a in cells]);row['partition_simplices']=len(allrays);row['geometry_counts']=g.counts.copy()
   for name,vs in (('accuracy',rays),('partition',allrays)):
    orders=np.full(len(vs),4);bounds=np.array([bernstein_bound(v,4) for v in vs]);nodes=len(vs)*4**(t-1)
    if nodes>300_000_000:row['integrals'].append(dict(name=name,skipped=True,nodes=nodes));continue
    value=grouped_mass(vs,orders,bounds,deadline=start+115);value['name']=name;row['integrals'].append(value);save();print(value,flush=True)
 except (ArithmeticError,TimeoutError) as e:
  row['error']=str(e);row['geometry_counts']=g.counts.copy()
  if hasattr(g,'last_failure'):
   v,n,msg=g.last_failure;np.savez_compressed(BASE/'latest_geometry_failure.npz',vertices=v,normal=n);row['qhull_error']=msg[:400]
 save()
save()

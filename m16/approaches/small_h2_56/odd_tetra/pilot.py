import json,time,resource,sys
from pathlib import Path
import numpy as np
from outer import grouped_mass,rule
BASE=Path(__file__).parent;PARENT=BASE.parent
sys.path.insert(0,str(PARENT))
from segment import grouped_mass as segments
from volume import bernstein_bound,supporting_lower
sys.path.pop(0)
start=time.process_time();out={'cases':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
for name in ('identity','t5_cones','t5_sigma0.0025_cones','t5_sigma0.02_cones'):
 if name=='identity':rays=np.array([np.eye(5)]);ids=np.array([0])
 else:
  allrays=np.load(PARENT/(name+'.npz'))['rays'];ids=np.linspace(0,len(allrays)-1,32,dtype=int);rays=allrays[ids]
 for n in (8,16):
  x,w=rule(n);assert abs(w.sum()-1/4)<1e-15
  bounds=np.array([bernstein_bound(v,n,degree=n-1,radial_lower=supporting_lower(v)) for v in rays]);orders=np.full(len(rays),n)
  row=dict(name=name,indices=ids.tolist(),order=n,tetra=grouped_mass(rays,orders,bounds,deadline=start+105),segment=segments(rays,orders,bounds,deadline=start+105,jacobi=True));row['difference']=abs(row['tetra']['estimate']-row['segment']['estimate'])
  if name=='identity':row['analytic_truth']=1/32;row['analytic_error']=abs(row['tetra']['estimate']-1/32)
  out['cases'].append(row);save();print(name,n,'difference',row['difference'],'counts',row['tetra']['primitive_counts'],'CPU',row['tetra']['cpu_seconds'],flush=True)
save()

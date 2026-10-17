import json,time,resource
from pathlib import Path
import numpy as np
from volume import mass,bernstein_bound,plan,grouped_mass
BASE=Path(__file__).parent;start=time.process_time();out={'primitive':[],'plans':[],'integrals':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle2.json').write_text(json.dumps(out,indent=2)+'\n')
for d in (4,5,6):
 for n in (4,8):
  v,b,k=mass(np.eye(d),n);bound=bernstein_bound(np.eye(d),n)
  assert abs(v-2**-d)<=bound
  out['primitive'].append(dict(d=d,order=n,error=abs(v-2**-d),bernstein_bound=bound));save()
for name in ('t5_cones','t5_partition'):
 rays=np.load(BASE/(name+'.npz'))['rays'];orders,bounds=plan(rays,1e-4)
 row=dict(name=name,target_bound=1e-4,actual_bound=float(bounds.sum()),histogram={str(n):int(np.sum(orders==n)) for n in np.unique(orders)},nodes=int(sum(int(n)**4 for n in orders)))
 out['plans'].append(row);save();print(row,flush=True)
 result=grouped_mass(rays,orders,bounds,deadline=start+115);result['name']=name
 out['integrals'].append(result);save();print(result,flush=True)
save()

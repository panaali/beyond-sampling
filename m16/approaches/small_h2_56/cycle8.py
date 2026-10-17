import json,time,resource
from pathlib import Path
import numpy as np
from segment import outer_rule,grouped_mass
from volume import plan
from volume import bernstein_bound
BASE=Path(__file__).parent;start=time.process_time();out={'primitive':[],'integrals':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle8.json').write_text(json.dumps(out,indent=2)+'\n')
for n in (4,8):
 y,rest,w=outer_rule(n,True);assert abs(w.sum()-1/24)<1e-16
 row=grouped_mass(np.array([np.eye(5)]),np.array([n]),np.array([bernstein_bound(np.eye(5),n,degree=n-1)]),jacobi=True);row.update(order=n,truth=1/32,error=abs(row['estimate']-1/32));out['primitive'].append(row);save()
for name in ('t5_cones','t5_partition','t5_sigma0.0025_cones','t5_sigma0.02_cones','crossseed1_cones','crossseed2_cones','crossseed3_cones','crossseed4_cones'):
 rays=np.load(BASE/(name+'.npz'))['rays'];s=time.process_time();orders,bounds=plan(rays,1e-4,jacobi=True);planning=time.process_time()-s
 s=time.process_time();row=grouped_mass(rays,orders,bounds,deadline=start+115,jacobi=True);row.update(name=name,planning_cpu_seconds=planning,integration_cpu_seconds=time.process_time()-s)
 out['integrals'].append(row);save();print(row,flush=True)
save()

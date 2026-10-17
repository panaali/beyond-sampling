import json,time,resource
from pathlib import Path
import numpy as np
from geometry import Geometry
from volume import bernstein_bound
from segment6 import grouped_mass,block
from models import load
BASE=Path(__file__).parent;start=time.process_time();out={'primitive':[],'partition':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle7.json').write_text(json.dumps(out,indent=2)+'\n')
for n in (4,8):
 y,r,w=block(n,0,n**4);assert abs(w.sum()-1/120)<1e-16
 bound=bernstein_bound(np.eye(6),n,degree=n-1);row=grouped_mass(np.array([np.eye(6)]),np.array([n]),np.array([bound]));row.update(truth=1/64,order=n,error=abs(row['estimate']-1/64));out['primitive'].append(row);save();print('primitive',row,flush=True)
model,sha=load(6);g=Geometry(20);cells=g.gates(model);rays=g.simplices([v for v,A in cells]);out['geometry_counts']=g.counts;out['checkpoint_sha256']=sha
np.savez_compressed(BASE/'t6_partition.npz',rays=rays);save()
for n in (4,8):
 bounds=np.array([bernstein_bound(v,n,degree=n-1) for v in rays]);row=grouped_mass(rays,np.full(len(rays),n),bounds,deadline=start+115);row['order']=n;out['partition'].append(row);save();print('partition',row,flush=True)
save()

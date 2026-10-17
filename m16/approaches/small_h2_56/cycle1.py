"""Frozen recovery diagnostics; output written after each completed stage."""
import json,time,resource
from pathlib import Path
import numpy as np
from geometry import Geometry
from volume import mass,sum_mass
from models import load,sample
BASE=Path(__file__).parent
start=time.process_time();out={'primitive':[],'models':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle1.json').write_text(json.dumps(out,indent=2)+'\n')
for d in (4,5,6):
 for n in (4,8):
  p,b,k=mass(np.eye(d),n)
  out['primitive'].append(dict(d=d,order=n,estimate=p,truth=2.**-d,error=abs(p-2.**-d),bound=b,nodes=k));save()
for t in (5,6):
 model,sha=load(t);row={'T':t,'sigma':0,'sha256':sha};out['models'].append(row)
 g=Geometry(15)
 try:
  cells=g.gates(model);row['gate_count']=len(cells);save()
  correct=g.correct(model,cells);rays=g.simplices(correct);row['correct_count']=len(correct);row['simplex_count']=len(rays);save()
  if t==5:
   np.savez_compressed(BASE/'t5_cones.npz',rays=rays)
   row['correct_integrals']=[]
   for n in (4,8):
    s=time.process_time();r=sum_mass(rays,n);r['cpu_seconds']=time.process_time()-s;row['correct_integrals'].append(r);save()
   allrays=g.simplices([v for v,a in cells]);row['partition_simplex_count']=len(allrays)
   np.savez_compressed(BASE/'t5_partition.npz',rays=allrays)
   row['partition_integral']=sum_mass(allrays,8);save()
 except (TimeoutError,ArithmeticError) as e:row['geometry_error']=str(e)
 row['geometry_counts']=g.counts;row['validation']=sample(model);save();print(json.dumps(row),flush=True)
save()

"""Fixed-tolerance cross-training-seed replication, never selected by outcome."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from geometry import Geometry
from volume import plan
from segment import grouped_mass
from models import load,sample
BASE=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('seeds',nargs='+',type=int);a=p.parse_args();start=time.process_time();out={'models':[]}
path=BASE/('crossseed_'+'_'.join(map(str,a.seeds))+'.json')
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 path.write_text(json.dumps(out,indent=2)+'\n')
for seed in a.seeds:
 model,sha=load(5,seed);row=dict(H=2,T=5,seed=seed,sha256=sha,tolerance=1e-4,integrals=[]);out['models'].append(row);save();g=Geometry(25);setup=time.process_time()
 try:
  cells=g.gates(model);correct=g.correct(model,cells);rays=g.simplices(correct);partition=g.simplices([v for v,A in cells]);row.update(geometry_counts=g.counts.copy(),setup_cpu_seconds=time.process_time()-setup,correct_simplices=len(rays),partition_simplices=len(partition));save();print('geometry',seed,row['geometry_counts'],flush=True)
  if len(rays)<=200000:np.savez_compressed(BASE/f'crossseed{seed}_cones.npz',rays=rays)
  for name,vs in (('accuracy',rays),('partition',partition)):
   before=time.process_time();orders,bounds=plan(vs,1e-4);planning=time.process_time()-before;before=time.process_time()
   value=grouped_mass(vs,orders,bounds,deadline=start+115);value.update(name=name,planning_cpu_seconds=planning,integration_cpu_seconds=time.process_time()-before);row['integrals'].append(value);save();print('integral',seed,value,flush=True)
 except (ArithmeticError,TimeoutError) as e:row['error']=str(e);row['geometry_counts']=g.counts.copy()
 if time.process_time()-start<115:row['validation']=sample(model)
 else:row['validation_skipped']='cycle CPU cap'
 save()
save()

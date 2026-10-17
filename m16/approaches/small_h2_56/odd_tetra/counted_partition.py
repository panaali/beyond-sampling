import json,resource,sys,time
from pathlib import Path
from counted import Audit,enumerate_correct,volume
BASE=Path(__file__).parent
sys.path.insert(0,str(BASE.parent))
from models import load,noisy
sys.path.pop(0)
start=time.process_time();base,sha=load(5);out={'models':[],'checkpoint_sha256':sha}
for sigma in (0.,.0025,.02):
 audit=Audit();model=base if sigma==0 else noisy(base,sigma);t=time.process_time()
 row={'sigma':sigma}
 try:
  _,geometry,rays=enumerate_correct(model,audit,partition=True)
  value,stats=volume(rays,audit)
  row.update(estimate=value,error=abs(value-1),bound=stats['conditional_outer_bound']+stats['positive_inner_bound'],integration=stats,
             gate_simplices=len(rays),geometry=geometry,cost=audit.report(),cpu_seconds=time.process_time()-t)
  assert row['error']<=row['bound']
 except Exception as exc:row.update(failure=repr(exc),cost=audit.report(),cpu_seconds=time.process_time()-t)
 out['models'].append(row);out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'counted_partition.json').write_text(json.dumps(out,indent=2)+'\n')
 print(sigma,row.get('estimate'),row.get('error'),row.get('bound'),row.get('failure'),flush=True)

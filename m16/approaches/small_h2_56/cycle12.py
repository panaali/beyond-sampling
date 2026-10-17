import json,time,resource
from pathlib import Path
import numpy as np
from triangle_outer import grouped_mass
BASE=Path(__file__).parent;FILE=BASE/'cycle12.json';start=time.process_time()
rays=np.load(BASE/'t6_partition.npz')['rays'];plan=np.load(BASE/'t6_partition_allocation_p3.npz');orders=plan['orders'];bounds=plan['bounds']
out=json.loads(FILE.read_text()) if FILE.exists() else {'chunks':[],'completed_cones':0,'cycles':[]}
cycle=dict(start_cone=out['completed_cones']);out['cycles'].append(cycle)
def save():
 cycle['cpu_seconds']=time.process_time()-start;cycle['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 out['complete']=out['completed_cones']==len(rays)
 out['estimate']=sum(c['estimate'] for c in out['chunks'])
 out['conditional_outer_quadrature_bound']=sum(c['conditional_outer_quadrature_bound'] for c in out['chunks'])
 out['inner_positive_quadrature_bound']=sum(c['inner_positive_quadrature_bound'] for c in out['chunks'])
 out['inner_series_truncation_bound']=sum(c['inner_series_truncation_bound'] for c in out['chunks'])
 out['total_conditional_integration_bound']=out['conditional_outer_quadrature_bound']+out['inner_positive_quadrature_bound']+out['inner_series_truncation_bound']
 if out['complete']:out['partition_error']=abs(out['estimate']-1)
 FILE.write_text(json.dumps(out,indent=2)+'\n')
while out['completed_cones']<len(rays) and time.process_time()-start<100:
 first=out['completed_cones'];last=first;nodes=0
 while last<len(rays) and nodes+int(orders[last])**3<=2000000:nodes+=int(orders[last])**3;last+=1
 try:row=grouped_mass(rays[first:last],orders[first:last],bounds[first:last],deadline=start+115)
 except Exception as exc:
  cycle['failure']=dict(first=first,last=last,error=repr(exc));save();raise
 row.update(first_cone=first,last_cone=last,nominal_triangle_calls=nodes);out['chunks'].append(row);out['completed_cones']=last;save();print('partition checkpoint',last,'/',len(rays),'mass',out['estimate'],'cpu',cycle['cpu_seconds'],flush=True)
save()

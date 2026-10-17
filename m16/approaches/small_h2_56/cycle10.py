import json,time,resource
from pathlib import Path
import numpy as np
from volume import bernstein_bound,supporting_lower
from allocation import bound_table,allocate,ORDERS
BASE=Path(__file__).parent;start=time.process_time();out={'cases':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle10.json').write_text(json.dumps(out,indent=2)+'\n')
for name in ('t5_cones','crossseed2_cones','t6_partition','t6_retained_cones'):
 rays=np.load(BASE/(name+'.npz'))['rays'];table,stats=bound_table(rays);d=rays.shape[1]
 # Compare vectorized bound computation against the scalar implementation.
 relative=[]
 for i in np.linspace(0,len(rays)-1,8,dtype=int):
  for j,n in enumerate(ORDERS):
   b=bernstein_bound(rays[i],int(n),degree=int(n)-1,radial_lower=supporting_lower(rays[i]))
   relative.append(abs(table[i,j]-b)/max(abs(b),1e-300))
 assert max(relative)<1e-11
 omitted=5.003269270003126e-6 if name=='t6_retained_cones' else 0.
 inner=1e-6 if d==6 else 0.
 row=dict(name=name,table_setup=stats,scalar_bound_max_relative_discrepancy=max(relative),pruned_mass_bound=omitted,inner_error_reserved=inner,allocations=[])
 for power in ((3,) if d==5 else (4,3)):
  orders,bounds,result=allocate(table,1e-4-omitted-inner,power)
  result['reserved_total_bound']=result['conditional_quadrature_error_bound']+omitted+inner
  _,_,matched=allocate(table,result['equal_allocation_bound'],power)
  result['matched_equal_rule_error_comparison']=matched
  row['allocations'].append(result)
  np.savez_compressed(BASE/(name+'_allocation_p'+str(power)+'.npz'),orders=orders,bounds=bounds)
 out['cases'].append(row);save();print(row,flush=True)
 if time.process_time()-start>115:raise TimeoutError('planning CPU guard')
save()

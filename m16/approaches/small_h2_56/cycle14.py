"""Preregistered cheaper T6 rule; reference read only after estimation finishes."""
import json,time,resource
from pathlib import Path
from math import gamma,pi,factorial
import numpy as np
from geometry import Geometry
from models import load
from allocation import bound_table,allocate
from triangle_outer import grouped_mass
BASE=Path(__file__).parent;FILE=BASE/'cycle14.json';start=time.process_time()
if FILE.exists():
 out=json.loads(FILE.read_text());rays=np.load(BASE/'cycle14_cold_rays.npz')['rays'];p=np.load(BASE/'cycle14_cold_plan.npz');orders=p['orders'];bounds=p['bounds']
else:
 model,sha=load(6);g=Geometry(25);cells=g.gates(model);pieces=g.correct(model,cells);rays=g.simplices(pieces);geometry_cpu=time.process_time()-start
 d=6;gram=np.swapaxes(rays,1,2)@rays;a=np.maximum((1-1e-12)/d,gram.min(axis=(1,2))-1e-12)
 upper=gamma(d/2)/(2*pi**(d/2)*factorial(d-1))*np.abs(np.linalg.det(rays))*a**(-d/2)
 small=upper<=1e-4/len(rays);omitted=float(upper[small].sum());original=len(rays);rays=rays[~small]
 table,table_stats=bound_table(rays);orders,bounds,allocation=allocate(table,1e-2-omitted-1e-6,3)
 if not allocation['feasible']:raise ArithmeticError('preregistered cheap table infeasible')
 out=dict(checkpoint_sha256=sha,total_planning_tolerance=1e-2,uses_reference_for_estimation=False,chunks=[],completed_cones=0,cycles=[],cold_setup=dict(cpu_seconds=time.process_time()-start,geometry_cpu_seconds=geometry_cpu,geometry_counts=g.counts,original_correct_simplices=original,retained_correct_simplices=len(rays),pruned_mass_bound=omitted,pruning_determinants=original,pruning_gram_arithmetic_ops=original*d*d*(2*d-1),table=table_stats,allocation=allocation,unmeasured_operations=['Qhull internals','clipping sorting and array bookkeeping','library determinant and power internals']),total_cost_audited=False)
 np.savez_compressed(BASE/'cycle14_cold_rays.npz',rays=rays);np.savez_compressed(BASE/'cycle14_cold_plan.npz',orders=orders,bounds=bounds)
cycle=dict(start_cone=out['completed_cones']);out['cycles'].append(cycle)
def save():
 cycle['cpu_seconds']=time.process_time()-start;cycle['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 out['complete']=out['completed_cones']==len(rays);out['estimate']=sum(c['estimate'] for c in out['chunks'])
 for key in ('conditional_outer_quadrature_bound','inner_positive_quadrature_bound','inner_series_truncation_bound'):out[key]=sum(c[key] for c in out['chunks'])
 out['total_conditional_error_bound']=out['conditional_outer_quadrature_bound']+out['inner_positive_quadrature_bound']+out['inner_series_truncation_bound']+out['cold_setup']['pruned_mass_bound']
 if out['complete']:
  reference=json.loads((BASE/'cycle13.json').read_text());assert reference['complete']
  difference=abs(out['estimate']-reference['estimate'])
  out['validation_only']=dict(reference_estimate=reference['estimate'],absolute_difference=difference,reference_conditional_error_bound=reference['total_conditional_error_bound'],conditional_error_ceiling=difference+reference['total_conditional_error_bound'],reference_used_for_estimation=False)
 FILE.write_text(json.dumps(out,indent=2)+'\n')
save();print('cold setup',out['cold_setup'],flush=True)
while out['completed_cones']<len(rays) and time.process_time()-start<100:
 first=out['completed_cones'];last=first;nodes=0
 while last<len(rays) and nodes+int(orders[last])**3<=2000000:nodes+=int(orders[last])**3;last+=1
 try:row=grouped_mass(rays[first:last],orders[first:last],bounds[first:last],deadline=start+115,inner_budget=1e-6,total_cones=len(rays))
 except Exception as exc:
  cycle['failure']=dict(first=first,last=last,error=repr(exc));save();raise
 row.update(first_cone=first,last_cone=last,nominal_triangle_calls=nodes);out['chunks'].append(row);out['completed_cones']=last;save();print('cheap checkpoint',last,'/',len(rays),'mass',out['estimate'],'cpu',cycle['cpu_seconds'],flush=True)
save()

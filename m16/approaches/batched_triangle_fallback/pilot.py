"""Frozen T106 partial pilot, local grouped callback only; ten alternating calls."""
import hashlib,json,math,time,statistics
from pathlib import Path
import numpy as np
from batch import Grouped,Engine,EDGE
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit
from m16.approaches.small_ladder_review.portable_cost.t6.jacobi import outer_rule
from m16.approaches.small_ladder_review.portable_cost.t6.geometry import determinant
HERE=Path(__file__).parent
CACHE=Path('/workspace/work/recover-small_h2_34/m16/approaches/small_ladder_review/portable_cost/t6')
class Mass:
 def __init__(self,audit,grouped):
  self.audit=audit;self.triangle=(Grouped if grouped else Engine)(audit);self.rules={}
 def one(self,ray,order,inner_budget):
  audit=self.audit;det=abs(determinant(ray,audit));factor=det/(math.pi*math.pi*math.pi);audit.add('mass_prefactor',10)
  if det==0:return 0.,0.,0.,dict(triangles=0,positive_quadrature_fallbacks=0,quadrature_nodes=0)
  if order not in self.rules:self.rules[order]=outer_rule(order,audit)
  y,rest,w=self.rules[order];prefix=y@ray[:,:3].T;frame=EDGE.prepare(ray[:,3:].T)
  total_weight=float(w.sum());maximum_weight=float(w.max());tolerance=inner_budget/(factor*total_weight)
  values,stats=EDGE.evaluate(prefix,rest,frame,self.triangle.positive,absolute_tolerance=tolerance,normalize_edges=False)
  value=factor*float(w@values)
  error=factor*min(total_weight*stats['maximum_quadrature_error_bound'],maximum_weight*stats['quadrature_error_bound'])
  series=factor*min(total_weight*stats['maximum_analytic_series_truncation_bound'],maximum_weight*stats['analytic_series_truncation_bound_sum'])
  audit.add('fresh_triangle_line_groups',1360*len(w));audit.add('frame_and_batch_setup',2560)
  audit.add('outer_endpoint_and_weight_arithmetic',80*len(w)+200)
  audit.events['triangles']+=len(w);audit.events['fixed_frames']+=1
  return value,error,series,stats

def main():
 start=time.process_time();expected={'projected_correct_rays.npz':'15ff556d1ef6168275d67a785eb7f2623a411cb8832816cf5cfc97060cd339c7','plan_1e-06.npz':'3126342f2705bf7db13483b603f423c0d9a036a02eb434b5628913850013ae81'}
 for name,sha in expected.items():assert hashlib.sha256((CACHE/name).read_bytes()).hexdigest()==sha
 rays=np.load(CACHE/'projected_correct_rays.npz')['rays'];orders=np.load(CACHE/'plan_1e-06.npz')['orders'];indices=np.linspace(0,len(rays)-1,64,dtype=int)
 def run(grouped):
  cpu=time.process_time();wall=time.perf_counter();audit=Audit();engine=Mass(audit,grouped);rows=[]
  for index in indices:
   value,error,series,stats=engine.one(rays[index],int(orders[index]),1e-8/len(rays))
   rows.append(dict(index=int(index),order=int(orders[index]),mass=value,inner_bound=error,series_bound=series,stats=stats))
  result=dict(cpu=time.process_time()-cpu,wall=time.perf_counter()-wall,rows=rows,cost=audit.report(),partial_mass=sum(r['mass'] for r in rows),inner_bound=sum(r['inner_bound'] for r in rows),series_bound=sum(r['series_bound'] for r in rows))
  if grouped:result.update(selection_cpu=engine.triangle.selection_cpu,node_cpu=engine.triangle.node_cpu,groups=engine.triangle.groups)
  return result
 out=dict(source_hashes=expected,indices=indices.tolist(),passes={},timings=[],full_accuracy_estimated=False,cost_independently_approved=False)
 for grouped in(False,True):out['passes']['grouped' if grouped else 'scalar']=run(grouped)
 pairs=list(zip(out['passes']['scalar']['rows'],out['passes']['grouped']['rows']))
 out['comparison']=dict(max_relative_mass_difference=max(abs(b['mass']/a['mass']-1) if a['mass'] else abs(b['mass']) for a,b in pairs),changed_stats_indices=[a['index'] for a,b in pairs if a['stats']!=b['stats']],max_absolute_mass_difference=max(abs(a['mass']-b['mass']) for a,b in pairs))
 assert not out['comparison']['changed_stats_indices']
 assert out['comparison']['max_relative_mass_difference']<1e-12
 # Exactly ten fresh complete calls total, five of each, alternating order.
 for i in range(10):
  grouped=bool(i%2);r=run(grouped);assert r['partial_mass']==out['passes']['grouped' if grouped else 'scalar']['partial_mass']
  out['timings'].append(dict(index=i,variant='grouped' if grouped else 'scalar',cpu=r['cpu'],wall=r['wall']))
 out['timing_medians']={name:statistics.median(r['cpu'] for r in out['timings'] if r['variant']==name) for name in ('scalar','grouped')}
 out['cpu_seconds']=time.process_time()-start
 (HERE/'pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ('passes','indices')},indent=2))
 for k,v in out['passes'].items():print(k,{x:v[x] for x in ('cpu','partial_mass','inner_bound','series_bound','cost')})
if __name__=='__main__':main()

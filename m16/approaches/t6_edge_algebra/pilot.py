"""Same64-cone partial integral for all three frozen implementations."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from check import HERE,Audit,Engine,outer_rule,prepare,evaluate,old
from m16.approaches.small_ladder_review.portable_cost.t6.geometry import determinant
CACHE=Path('/workspace/work/recover-small_h2_34/m16/approaches/small_ladder_review/portable_cost/t6')
CHARGE={'baseline':1648,'raw':1360,'normalized':1456}

class Mass:
 def __init__(self,audit,variant):
  self.audit=audit;self.triangle=Engine(audit);self.rules={};self.variant=variant
 def one(self,ray,order,inner_budget):
  audit=self.audit;det=abs(determinant(ray,audit));factor=det/(math.pi*math.pi*math.pi);audit.add('mass_prefactor',10)
  if det==0:return 0.,0.,0.,dict(triangles=0,positive_quadrature_fallbacks=0,quadrature_nodes=0)
  if order not in self.rules:self.rules[order]=outer_rule(order,audit)
  y,rest,w=self.rules[order];prefix=y@ray[:,:3].T;frame=prepare(ray[:,3:].T)
  total_weight=float(w.sum());maximum_weight=float(w.max());tolerance=inner_budget/(factor*total_weight)
  if self.variant=='baseline':values,stats=old.evaluate(prefix,rest,frame,self.triangle.positive,absolute_tolerance=tolerance)
  else:values,stats=evaluate(prefix,rest,frame,self.triangle.positive,absolute_tolerance=tolerance,normalize_edges=self.variant=='normalized')
  value=factor*float(w@values)
  error=factor*min(total_weight*stats['maximum_quadrature_error_bound'],maximum_weight*stats['quadrature_error_bound'])
  series=factor*min(total_weight*stats['maximum_analytic_series_truncation_bound'],maximum_weight*stats['analytic_series_truncation_bound_sum'])
  audit.add('fresh_triangle_line_groups',CHARGE[self.variant]*len(w))
  audit.add('frame_and_batch_setup',512+2048)
  audit.add('outer_endpoint_and_weight_arithmetic',80*len(w)+200)
  # Existing outer80/node overlaps some endpoints: retained conservatively.
  audit.events['triangles']+=len(w);audit.events['fixed_frames']+=1
  return value,error,series,stats

def main():
 start=time.process_time();expected={'projected_correct_rays.npz':'15ff556d1ef6168275d67a785eb7f2623a411cb8832816cf5cfc97060cd339c7','plan_1e-06.npz':'3126342f2705bf7db13483b603f423c0d9a036a02eb434b5628913850013ae81'}
 for name,sha in expected.items():assert hashlib.sha256((CACHE/name).read_bytes()).hexdigest()==sha
 rays=np.load(CACHE/'projected_correct_rays.npz')['rays'];orders=np.load(CACHE/'plan_1e-06.npz')['orders'];indices=np.linspace(0,len(rays)-1,64,dtype=int)
 out=dict(source_hashes=expected,indices=indices.tolist(),passes={},full_accuracy_estimated=False,cost_independently_approved=False)
 def save():
  out['cpu_seconds']=time.process_time()-start;(HERE/'pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
 for name in ('baseline','raw','normalized'):
  cpu=time.process_time();wall=time.perf_counter();audit=Audit();engine=Mass(audit,name);rows=[];out['passes'][name]=dict(rows=rows);cc=0.;cw=0.
  for index in indices:
   try:
    value,error,series,stats=engine.one(rays[index],int(orders[index]),1e-8/len(rays))
    rows.append(dict(index=int(index),order=int(orders[index]),mass=value,inner_bound=error,series_bound=series,stats=stats))
   except Exception as exc:out['failure']=dict(variant=name,index=int(index),type=type(exc).__name__,message=str(exc));save();raise
   c=time.process_time();t=time.perf_counter();save();cc+=time.process_time()-c;cw+=time.perf_counter()-t
  out['passes'][name].update(cpu=time.process_time()-cpu-cc,wall=time.perf_counter()-wall-cw,checkpoint_cpu=cc,checkpoint_wall=cw,cost=audit.report(),partial_mass=sum(r['mass'] for r in rows),inner_bound=sum(r['inner_bound'] for r in rows),series_bound=sum(r['series_bound'] for r in rows));save()
 out['comparisons']={}
 for name in ('raw','normalized'):
  pairs=list(zip(out['passes']['baseline']['rows'],out['passes'][name]['rows']))
  out['comparisons'][name]=dict(max_relative_mass_difference=max(abs(b['mass']/a['mass']-1) if a['mass'] else abs(b['mass']) for a,b in pairs),changed_fallback_indices=[a['index'] for a,b in pairs if a['stats']['positive_quadrature_fallbacks']!=b['stats']['positive_quadrature_fallbacks']])
 save();print(json.dumps(out['comparisons'],indent=2))
 for name,v in out['passes'].items():print(name,{k:v[k] for k in ('cpu','wall','partial_mass','inner_bound','series_bound')},'ops',v['cost']['known_arithmetic_upper_bound'])
if __name__=='__main__':main()

"""Unchanged T52 64-cone pilot with paired frame implementation."""
import hashlib,json,math,time
import numpy as np
from check import HERE,Audit,Engine,outer_rule,prepare,evaluate
from m16.approaches.small_ladder_review.portable_cost.t6.integration import MassEngine
from m16.approaches.small_ladder_review.portable_cost.t6.geometry import determinant
from pathlib import Path
CACHE=Path('/workspace/work/recover-small_h2_34/m16/approaches/small_ladder_review/portable_cost/t6')

class FixedMass:
 def __init__(self,audit):
  self.audit=audit;self.triangle=Engine(audit);self.rules={}
 def one(self,ray,order,inner_budget):
  audit=self.audit;det=abs(determinant(ray,audit));factor=det/(math.pi*math.pi*math.pi);audit.add('mass_prefactor',10)
  if det==0:return 0.,0.,0.,dict(triangles=0,positive_quadrature_fallbacks=0,quadrature_nodes=0)
  if order not in self.rules:self.rules[order]=outer_rule(order,audit)
  y,rest,w=self.rules[order];prefix=y@ray[:,:3].T
  total_weight=float(w.sum());maximum_weight=float(w.max());tolerance=inner_budget/(factor*total_weight)
  frame=prepare(ray[:,3:].T)
  values,stats=evaluate(prefix,rest,frame,self.triangle.positive,absolute_tolerance=tolerance)
  value=factor*float(w@values)
  error=factor*min(total_weight*stats['maximum_quadrature_error_bound'],maximum_weight*stats['quadrature_error_bound'])
  series=factor*min(total_weight*stats['maximum_analytic_series_truncation_bound'],maximum_weight*stats['analytic_series_truncation_bound_sum'])
  # Retain common legacy allowance, subtract only the identified frame block.
  audit.add('triangle_common_legacy_allowance',(10000-111)*len(w)+10000)
  audit.add('fixed_frame_arithmetic',111+3*len(w))
  audit.add('outer_endpoint_and_weight_arithmetic',80*len(w)+200)
  stats['setup_frames']=1
  return value,error,series,stats

def main():
 begin=time.process_time();paths={'rays':CACHE/'projected_correct_rays.npz','plan':CACHE/'plan_1e-06.npz'}
 expected={'rays':'15ff556d1ef6168275d67a785eb7f2623a411cb8832816cf5cfc97060cd339c7','plan':'3126342f2705bf7db13483b603f423c0d9a036a02eb434b5628913850013ae81'}
 for k,p in paths.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[k]
 data=np.load(paths['rays']);rays=data['rays'];orders=np.load(paths['plan'])['orders'];indices=np.linspace(0,len(rays)-1,64,dtype=int)
 out=dict(source_hashes=expected,indices=indices.tolist(),passes={},full_accuracy_estimated=False)
 def save():
  out['cpu_seconds']=time.process_time()-begin;(HERE/'pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
 for name,cls in (('old',MassEngine),('new',FixedMass)):
  c=time.process_time();t=time.perf_counter();audit=Audit();engine=cls(audit);rows=[];out['passes'][name]=dict(rows=rows);checkpoint_cpu=0.;checkpoint_wall=0.
  for index in indices:
   try:
    mass,inner,series,stats=engine.one(rays[index],int(orders[index]),1e-8/len(rays))
    rows.append(dict(index=int(index),order=int(orders[index]),mass=mass,inner_bound=inner,series_bound=series,stats=stats))
   except Exception as exc:out['failure']=dict(pass_name=name,index=int(index),type=type(exc).__name__,message=str(exc));save();raise
   cc=time.process_time();tt=time.perf_counter();save();checkpoint_cpu+=time.process_time()-cc;checkpoint_wall+=time.perf_counter()-tt
  out['passes'][name].update(cpu=time.process_time()-c-checkpoint_cpu,wall=time.perf_counter()-t-checkpoint_wall,checkpoint_cpu=checkpoint_cpu,checkpoint_wall=checkpoint_wall,cost=audit.report(),partial_mass=sum(r['mass'] for r in rows),inner_bound=sum(r['inner_bound'] for r in rows),series_bound=sum(r['series_bound'] for r in rows));save()
 pairs=list(zip(out['passes']['old']['rows'],out['passes']['new']['rows']))
 out['max_relative_mass_difference']=max(abs(b['mass']/a['mass']-1) if a['mass'] else abs(b['mass']) for a,b in pairs)
 out['changed_fallbacks']=[a['index'] for a,b in pairs if a['stats']['positive_quadrature_fallbacks']!=b['stats']['positive_quadrature_fallbacks']]
 out['diagnosis_required']=bool(out['max_relative_mass_difference']>1e-8 or out['changed_fallbacks'])
 save();print(json.dumps({k:v for k,v in out.items() if k not in ('passes','indices')},indent=2))
 for k,v in out['passes'].items():print(k,{q:v[q] for q in ('cpu','wall','partial_mass','inner_bound','series_bound')})
if __name__=='__main__':main()

"""Aggregate saved full-phase shards and their finite costs without rerunning cones."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,math,time
import numpy as np
P=Path(__file__).resolve().parent

def main():
 start=time.process_time();proto=json.loads((P/'protocol.json').read_text());rows=[];shards=[]
 for q in sorted(P.glob('shard_*.json.gz')):
  d=json.loads(gzip.decompress(q.read_bytes()));rows.extend(d['rows']);shards.append(dict(file=q.name,sha256=hashlib.sha256(q.read_bytes()).hexdigest(),complete=d['complete']))
 indices=[r['index'] for r in rows];assert len(indices)==len(set(indices)) and indices==sorted(indices)
 N=proto['N'];complete=len(rows)==N and indices==list(range(N)) and all(s['complete'] for s in shards)
 known=Counter();exact=Counter();methods=Counter();orders=Counter();stages=Counter();bits=[0,0];values=np.full(N,np.nan);bounds=np.full(N,np.nan);errors=[];exact_cpu=mp_cpu=0.;new_kernel_cpu=0.
 for r in rows:
  assert r['mode']!='failure' and all(r.get(k) is not False for k in ('comparison_pass','bound_pass','unchanged_numerical_result')),r['index']
  known.update(r['center']['setup_selection_counts']);known.update(r.get('repeated_determinant_counts',{}));known.update(r.get('kernel_counts',{}));methods[r.get('method',r['mode'])]+=1
  if r['mode']=='returned':
   values[r['index']]=r['value'];bounds[r['index']]=r['bound'];errors.append(r['error']);assert r['comparison_pass'] and r['bound']<=1e-7/N
   if r.get('order'):orders[r['order']]+=1
   elif 'rescue' in r and r['rescue']['order']:orders[r['rescue']['order']]+=1
  if 'rescue' in r:
   a=r['rescue']['exact_operation_inventory'];exact.update(a['operations']);bits[0]=max(bits[0],a['max_numerator_bits']);bits[1]=max(bits[1],a['max_denominator_bits'])
   for step in r['rescue']['stages']:stages[step['method']]+=1;exact_cpu+=step.get('certificate_cpu',0);mp_cpu+=step.get('selection_cpu',0)
   new_kernel_cpu+=r['rescue'].get('kernel_cpu',0)
 sessions=[json.loads(q.read_text()) for q in sorted(P.glob('session_*.json'))];pilot=json.loads((P.parent/'stable_moment_h8/pilot.json').read_text());rules={};all_rule_cost=Counter()
 for session in [pilot]+sessions:
  for e in session.get('cold_rule_events',session.get('cold_rules',[])):
   order=e['order'];counts=e['counts'];all_rule_cost.update(counts)
   if order in rules:assert counts==rules[order]
   else:rules[order]=counts
 canonical=Counter()
 for counts in rules.values():canonical.update(counts)
 known.update(canonical);known['universal_constant_arithmetic']+=6
 out=dict(complete_corpus=complete,attempted=len(rows),returned=int(np.isfinite(values).sum()),unresolved_indices=[r['index'] for r in rows if r['mode']!='returned'],reused_pilot_rows=sum(r.get('reused_pilot',False) for r in rows),method_counts=dict(methods),quadrature_orders=dict(orders),exact_stage_counts=dict(stages),known_float_named_counts=dict(known),known_float_named_total=sum(known.values()),canonical_cold_rule_counts=dict(canonical),canonical_rule_orders=sorted(rules),actual_staged_rule_counts=dict(all_rule_cost),covered_error_sum=math.fsum(errors),covered_absolute_error_sum=math.fsum(map(abs,errors)),maximum_absolute_error=max(map(abs,errors),default=0),covered_conditional_bound_sum=math.fsum(bounds[np.isfinite(bounds)]),exact_operation_inventory=dict(exact),max_rational_numerator_bits=bits[0],max_rational_denominator_bits=bits[1],reported_exact_certificate_cpu=exact_cpu,reported_mp_selection_cpu=mp_cpu,total_staged_cpu_including_imports=pilot['cpu_including_imports']+math.fsum(s['cpu'] for s in sessions),shards=shards,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),complete_implementation_cost=False,full_floating_certificate=False,scope='Conditional numerical reference/known-category audit on unchanged captured geometry. Exact integer/gcd/MP and Legendre internal costs separate; memory/control/native uncertainty retained. No full-rung/noise or sampling claim.')
 bymethod=Counter()
 for r in rows:bymethod[r.get('method',r['mode'])]+=sum(r['center']['setup_selection_counts'].values())+sum(r.get('kernel_counts',{}).values())+sum(r.get('repeated_determinant_counts',{}).values())
 out['known_row_cost_by_method']=dict(bymethod)
 out.update(moment_attempts=sum('moment' in r['center'] for r in rows),moment_failures=[dict(index=r['index'],reason=r['center']['moment']['failure']) for r in rows if 'failure' in r['center'].get('moment',{})],unchanged_nonmoment_rows=sum(r.get('unchanged_numerical_result',False) for r in rows),pilot_exact_validation_cpu=pilot['exact_validation_cpu'])
 baseline=json.loads((Path(proto['baseline_folder'])/'summary.json').read_text());out.update(baseline_known_total=baseline['known_float_named_total'],known_cost_ratio=out['known_float_named_total']/baseline['known_float_named_total'],known_unit_saving=baseline['known_float_named_total']-out['known_float_named_total'])
 if complete and np.isfinite(values).all():
  out.update(error_mass=math.fsum(values),accuracy=1-math.fsum(values),reference_accuracy=.9761608075466027,accuracy_minus_reference=1-math.fsum(values)-.9761608075466027,average_known_units_per_cone=sum(known.values())/N,original_t109_300_units_goal_met=sum(known.values())/N<300)
 out['aggregation_cpu']=time.process_time()-start
 (P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(P/'values.npz',values=values,bounds=bounds)
 print(json.dumps({k:v for k,v in out.items() if k not in ('shards','exact_operation_inventory','known_float_named_counts','unresolved_indices')},indent=2))
if __name__=='__main__':main()

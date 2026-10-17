"""Exactly the frozen64 previous cones; unchanged references and all costs retained."""
import time
START=time.process_time()
from pathlib import Path
from collections import Counter
import hashlib,json,math,resource
import numpy as np
import variant
P=Path(__file__).resolve().parent

def counts(row):
 c=Counter(row['center']['setup_selection_counts']);c.update(row.get('kernel_counts',{}));c.update(row.get('repeated_determinant_counts',{}));return c

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));M,old,kernel,rescue,radial,moment,source=variant.load();rules=kernel.Rules();baseline=json.loads((P.parent/'positive_h8_cost/pilot.json').read_text());bases={r['index']:r for r in baseline['rows']};rays=np.load(M.REF/'error_captured.npz')['rays'];assert len(rays)==variant.PROTOCOL['N'];assert list(bases)==variant.PROTOCOL['indices'];basecounts=Counter()
 for r in bases.values():basecounts.update(counts(r))
 basecounts.update(baseline['cold_rule_counts']);basecounts['universal_constant_arithmetic']+=6;assert sum(basecounts.values())==baseline['known_float_named_total']
 out=dict(complete=False,rows=[],baseline_known_total=sum(basecounts.values()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),variant_sha256=hashlib.sha256((P/'variant.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest(),conditional_only=True,full_cost=False,full_floating_certificate=False)
 def save():
  c=Counter()
  for r in out['rows']:c.update(counts(r))
  c.update(rules.ledger.dict());c['universal_constant_arithmetic']+=6
  out.update(known_counts=dict(c),known_total=sum(c.values()),cold_rules=rules.events,cold_rule_counts=rules.ledger.dict(),cpu_including_imports=time.process_time()-START)
  (P/'pilot.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 for index in variant.PROTOCOL['indices']:
  if time.process_time()-START>18:out['stop_reason']='18CPU cap';save();return
  base=bases[index];row=variant.one(rays[index],M,old,kernel,rescue,radial,rules);row.update(index=index,reference=base['reference'],reference_bound=base['reference_bound'],baseline_value=base['value'],baseline_method=base['method'],baseline_known_row_total=sum(counts(base).values()))
  if row['mode']=='returned':
   row.update(error=row['value']-row['reference'],comparison_pass=abs(row['value']-row['reference'])<=row['bound']+row['reference_bound']+5e-14)
   assert row['bound']<=M.TOL
   if row['method']!='moment_interval':row['unchanged_numerical_result']=row['value']==base['value'] and row['bound']==base['bound'];assert row['unchanged_numerical_result']
  out['rows'].append(row);save()
  if row['mode']=='failure' or row.get('comparison_pass') is False:out['stop_reason']='Explicit failure/reference gate';save();return
 out.update(complete=True,methods=dict(Counter(r.get('method',r['mode']) for r in out['rows'])),moment_attempts=sum('moment' in r['center'] for r in out['rows']),moment_failures=[dict(index=r['index'],reason=r['center']['moment']['failure']) for r in out['rows'] if 'failure' in r['center'].get('moment',{})],unresolved_indices=[r['index'] for r in out['rows'] if r['mode']!='returned'])
 good=[r for r in out['rows'] if r['mode']=='returned'];out.update(error_sum=math.fsum(r['error'] for r in good),absolute_error_sum=math.fsum(abs(r['error']) for r in good),conditional_bound_sum=math.fsum(r['bound'] for r in good));save();out.update(known_cost_ratio=out['known_total']/out['baseline_known_total'],known_unit_saving=out['baseline_known_total']-out['known_total']);save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','cold_rules')},indent=2))
if __name__=='__main__':main()

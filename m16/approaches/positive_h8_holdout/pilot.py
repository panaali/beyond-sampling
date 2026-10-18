"""Unchanged positive-mass selector, frozen seed1 geometry and fixed64 gate."""
import time
START=time.process_time()
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib,importlib.util,json,math,resource
import numpy as np
P=Path(__file__).resolve().parent
PROTOCOL=json.loads((P/'protocol.json').read_text())
REF=Path('/workspace/work/recover-small_h2_910/m16/approaches/frozen_recipe_holdout/exact')

def load():
 for f,h in PROTOCOL['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 path=P.parent/'positive_h8_cost/run.py';spec=importlib.util.spec_from_file_location('seed1_unchanged_selector',path);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
 old,kernel,rescue,radial=M.load();N=PROTOCOL['N'];M.TOL=1e-7/N;rescue.TOLERANCE=Fraction(1,10**7*N)
 assert Fraction(M.TOL)<=rescue.TOLERANCE and abs(M.TOL-float(rescue.TOLERANCE))<=math.ulp(float(rescue.TOLERANCE))
 rays=np.load(REF/'error_captured.npz')['rays'];assert len(rays)==N
 reference=np.full(N,np.nan);bounds=np.full(N,np.nan);seen=np.zeros(N,int)
 for i in range(8):
  a=np.load(REF/f'batch_{i}.npz');reference[a['indices']]=a['positive'];bounds[a['indices']]=a['bounds'];seen[a['indices']]+=1
 assert np.all(seen==1) and abs(1-math.fsum(reference)-.9874600108260589)<2e-16
 return M,old,kernel,rescue,radial,rays,reference,bounds

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));M,old,kernel,rescue,radial,rays,reference,bounds=load();rules=kernel.Rules();out=dict(N=PROTOCOL['N'],indices=PROTOCOL['indices'],float_tolerance=M.TOL,exact_tolerance=str(rescue.TOLERANCE),rows=[],complete=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest())
 def save():
  c=Counter()
  for r in out['rows']:c.update(r['center']['setup_selection_counts']);c.update(r.get('repeated_determinant_counts',{}));c.update(r.get('kernel_counts',{}))
  c.update(rules.ledger.dict());c['universal_constant_arithmetic']+=6
  out.update(cpu=time.process_time()-START,known_float_named_counts=dict(c),known_float_named_total=sum(c.values()),cold_rule_events=rules.events,cold_rule_counts=rules.ledger.dict(),exact_setup_in_known_float_total=False,complete_cost=False,full_floating_certificate=False,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  (P/'pilot.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 for index in PROTOCOL['indices']:
  if time.process_time()-START>18:out['stop_reason']='18CPU soft cap';break
  row=M.one(rays[index],old,kernel,rescue,radial,rules);row.update(index=index,reference=float(reference[index]),reference_bound=float(bounds[index]))
  if row['mode']=='returned':row.update(error=row['value']-row['reference'],comparison_pass=bool(abs(row['value']-row['reference'])<=row['bound']+row['reference_bound']+5e-14));assert row['bound']<=M.TOL
  out['rows'].append(row);save()
  if row['mode']=='failure' or row.get('comparison_pass') is False:out['stop_reason']='Explicit failure/comparison gate';break
 else:out['complete']=True
 good=[r for r in out['rows'] if r['mode']=='returned'];out.update(returned=len(good),methods=dict(Counter(r.get('method',r['mode']) for r in out['rows'])),unresolved_indices=[r['index'] for r in out['rows'] if r['mode']!='returned'],error_sum=math.fsum(r['error'] for r in good),absolute_error_sum=math.fsum(abs(r['error']) for r in good),maximum_absolute_error=max((abs(r['error']) for r in good),default=0),conditional_bound_sum=math.fsum(r['bound'] for r in good))
 save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','indices','cold_rule_events')},indent=2))
if __name__=='__main__':main()

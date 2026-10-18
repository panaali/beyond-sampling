"""Saved-only T196 budget bridge. No estimator, mass or model imports."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time,sys
HERE=Path(__file__).resolve().parent
C=Counter()
def cs(H,T):
 assert H>0 and T>=2
 # Adopted protocol count, not measured native instruction price.
 return T*(2*H*H+2*H)+2*H*T+T*H+3*T-2

def threshold(w,c):
 assert isinstance(w,F) and 0<w<=1 and isinstance(c,int) and c>0
 q=1/(4*w*w);C['rational_multiply']+=2;C['rational_divide']+=1
 n=q.numerator//q.denominator+1;B=n*c
 C['integer_floor_divide']+=1;C['integer_add']+=1;C['integer_multiply']+=1
 assert 4*n*w*w>1
 assert n==1 or 4*(n-1)*w*w<=1
 assert B//c==n and (B-1)//c==n-1
 return {'width_lower':str(w),'sample_cost_T196':c,'minimal_samples':n,'minimal_integer_budget':B,
  'strict_se_check_squared':str(4*n*w*w),'previous_sample_check_squared':str(4*(n-1)*w*w),
  'budget_theorem':'For every integer B>=minimal_integer_budget, floor(B/c) samples have maximal Bernoulli SD strictly below this certificate-width lower bound.'}

def main():
 start=time.process_time();sys.set_int_max_str_digits(0)
 out=HERE/'budget_results.json';progress=HERE/'budget_progress.jsonl'
 assert not out.exists() and not progress.exists()
 P=json.loads((HERE/'budget_protocol.json').read_text())
 for path,h in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 data=json.loads((HERE/'recovery_results.json').read_text());assert data['complete'] and data['saved_pair_checks']==72
 result={'complete':False,'controls':[],'models':[],'protocol_sha256':hashlib.sha256((HERE/'budget_protocol.json').read_bytes()).hexdigest(),
  'network_forwards':0,'mass_calls':0,'reference_quadrature_calls':0,'random_draws':0,'native_counter_calls':0,
  'scope':'Canonical common-budget certificate-width versus maximal sampling SD only; no actual-error/MSE lower bound or candidate implementation cost/feasibility claim.'}
 def record(r):
  with progress.open('a') as f:f.write(json.dumps(r)+'\n')
 def save():
  result.update(named_bridge_counts=dict(C),work_cpu_seconds=time.process_time()-start,process_cpu_seconds=time.process_time())
  out.write_text(json.dumps(result,indent=2)+'\n')
 try:
  for w,c,n,B in P['controls']:
   record({'stage':'control_attempt','w':w,'c':c});r=threshold(F(w),c)
   assert (r['minimal_samples'],r['minimal_integer_budget'])==(n,B)
   result['controls'].append(r);record({'stage':'control_result','row':r});save()
  assert len(data['models'])==2
  for m,H,expected_cost in zip(data['models'],[4,8],[218,682],strict=True):
   assert m['name']==f'H{H}T4seed0'
   record({'stage':'model_attempt','name':m['name']})
   c=cs(H,4);assert c==expected_cost
   r=threshold(F(m['integrated_accuracy_interval_width_lower']),c)
   assert r['minimal_samples']==m['sampling_SE_threshold']['samples_at_or_above']
   r.update(name=m['name'],certified_boxes=m['certified_boxes'],source_mass=m['certified_union_mass'])
   result['models'].append(r);record({'stage':'model_result','row':r});save()
  result['complete']=True;save()
  print(json.dumps([{'name':r['name'],'cost':r['sample_cost_T196'],'budget':r['minimal_integer_budget']} for r in result['models']]))
 except BaseException as e:
  result['failure']=repr(e);record({'stage':'failure','error':repr(e)});save();raise
if __name__=='__main__':main()

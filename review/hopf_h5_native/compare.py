"""Exact saved-float one-case comparison under original75-family truth box."""
import hashlib,json,resource,time
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def budget(C,per,batch):
 def cost(n):return per*n+batch*((n+999999)//1000000)+2
 n=max(0,(C-2)//per)
 while cost(n)>C:n-=1
 while cost(n+1)<=C:n+=1
 assert n>0 and cost(n)<=C<cost(n+1)
 return n,cost(n)
def ratio(n,e,p):return n*(e-p)**2/(p*(1-p))
def encode(q):return {'fraction':str(q),'float':float(q)}
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));dst=HERE/'comparison.json';assert not dst.exists()
 out={'complete':False,'rows':[],'source_sha256':sha(__file__)};start=time.process_time()
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  scope=json.loads((HERE/'compare_scope.json').read_text())
  for p,h in scope['pins'].items():assert sha(p)==h,p
  pilot=json.loads((HERE/'pilot.json').read_text());assert pilot['complete']
  sampler=json.loads((HERE/'sampler_control.json').read_text());assert sampler['complete']
  models=json.loads((HERE/'models.json').read_text());saved=next(r for r in models['models'] if (r['seed'],r['copy'])==(0,0));assert pilot['model_sha256']==saved['model_sha256']
  audit=json.loads(Path(scope['truth_box_file']).read_text());assert audit['family_size']==75
  box=next(r for r in audit['mc_truth_boxes'] if (r['H'],r['T'],r['seed'],r['index'])==(5,4,0,0))
  assert box['n']==saved['mc_n'] and box['hits']==saved['mc_count_inferred_from_report']
  p=F(box['hits'],box['n']);e=F(pilot['value']);lo,hi=F(box['lower']),F(box['upper']);assert 0<lo<p<hi<1
  out.update(truth_box=box,truth_box_family_size=75,nominal_joint_coverage=audit['nominal_joint_coverage'],numerical_beta_not_outward_rounded=audit['numerical_beta_not_outward_rounded'],scope='Single original H5 model; nominal75-family10M CP box, no new truth or h128-as-truth. Candidate source allowance is conditional, not full native/runtime ceiling. A ratio above1 for an upper-cost proxy is not an actual-cost failure.')
  C=pilot['conditional_source_with_constant_setup']
  for name,per,ctl in [('dense_original',281,8),('analytic_initial_zero',231,7)]:
   n,cost=budget(C,per,ctl);rs=[ratio(n,e,q) for q in (lo,hi)];bounds=[F(0) if lo<=e<=hi else min(rs),max(rs)]
   row={'sampler':name,'candidate_conditional_source_budget':C,'per_sample_with_aggregation':per,'per_batch_units':ctl,'samples':n,'spent':cost,'estimate':float(e),'truth_report':float(p),'error':encode(e-p),'point_ratio':encode(ratio(n,e,p)),'nominal_truth_box_ratio_range':[encode(x) for x in bounds]};out['rows'].append(row);save()
  out['complete']=True
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps({'complete':out['complete'],'ratios':[[r['point_ratio']['float'],r['nominal_truth_box_ratio_range'][1]['float']] for r in out['rows']]}))
if __name__=='__main__':main()

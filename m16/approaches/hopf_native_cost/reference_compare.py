"""Saved same-checkpoint comparison, conditional source budget and reference only."""
import hashlib,json
from decimal import Decimal,localcontext
from pathlib import Path
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_34/m16/approaches')
REF=AUTHOR/'radial_t4_reference/combined.json';RUNNER=AUTHOR/'class_region_t4/runner.py'

def D(x):return Decimal.from_float(float(x))
def budget_samples(budget,per_sample,per_batch,chunk=1000000):
 def cost(n):return per_sample*n+per_batch*((n+chunk-1)//chunk)+2
 n=max((budget-2)//per_sample,0)
 while cost(n)>budget:n-=1
 while cost(n+1)<=budget:n+=1
 assert n>0 and cost(n)<=budget<cost(n+1)
 return n,cost(n)
def ratio(n,estimate,p):return Decimal(n)*(estimate-p)**2/(p*(1-p))
def main():
 dst=HERE/'reference_comparison.json';assert not dst.exists()
 reference=json.loads(REF.read_text());assert reference['cones']==6347 and not reference['full_float_accuracy_certified']
 identity=json.loads((HERE/'model_identity.json').read_text());assert identity['checkpoint_sha256'] in RUNNER.read_text()
 h8=json.loads((HERE/'model_probe_summary.json').read_text());h64=json.loads((HERE/'model_h64_summary.json').read_text())
 p=D(reference['estimate']);b=D(reference['integration_only_bound']);out={'claim':'73cb0f00','reference':reference,'reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(RUNNER.read_bytes()).hexdigest(),'model_sha256':identity['float64_c_order_sha256'],'checkpoint_sha256':identity['checkpoint_sha256'],'reference_independent_review':'449a01e','rows':[],'scope':'Single clean model; reference integration-only bound excludes floating geometry/kernel error. Budget is declared expression subset plus conditional source allowance, not full runtime. No noise-panel or rung claim.'}
 with localcontext() as ctx:
  ctx.prec=60
  lo,hi=p-b,p+b
  for level,row in [('h8',h8['arms'][0]),('h64',h64['arms'][0])]:
   e=D(row['value']);c=row['conditional_source_allowance_total'] if level=='h8' else row['conditional_source_with_constant_setup']
   for name,per,control in [('dense_original',193,8),('analytic_zero_state_optimized',161,7)]:
    n,spent=budget_samples(c,per,control);bounds=[ratio(n,e,x) for x in (lo,hi)];minimum=Decimal(0) if lo<=e<=hi else min(bounds)
    out['rows'].append({'level':level,'sampler':name,'per_sample_with_hit_aggregation':per,'per_batch_index_control':control,'candidate_conditional_source_budget':c,'sampling_draws':n,'sampling_explicit_cost':spent,'estimate':float(e),'error':str(e-p),'conditional_error_interval':[str(e-hi),str(e-lo)],'matched_sampling_variance':str(p*(1-p)/Decimal(n)),'conditional_point_mse_ratio':str(ratio(n,e,p)),'conditional_integration_only_ratio_range':[str(minimum),str(max(bounds))],'note':'The optimized comparator is an explicitly derived first-zero-state implementation, not the unchanged zoo44 source.' if name.startswith('analytic') else 'Actual zoo44 dense indicator count plus hit aggregation, final division and explicit loop/index controls; RNG and true labels free.'})
  import tracer
  saved=json.loads((tracer.ROOT/'frozen/ev_44.json').read_text())['0']['rows'][0]
  out['saved_h128_reference_error_against_positive']=str(D(saved['ref'])-p)
  out['old_h64_reference_error']=str(D(saved['h64'])-D(saved['ref']))
 dst.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':out['rows'],'h128_reference_error':out['saved_h128_reference_error_against_positive']}))
if __name__=='__main__':main()

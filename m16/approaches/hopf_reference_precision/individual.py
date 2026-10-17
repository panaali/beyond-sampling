"""All25 individual models, with the unchanged simultaneous truth family."""
import hashlib,json,sys
from fractions import Fraction as Q
from pathlib import Path
HERE=Path(__file__).resolve().parent;COST=HERE.parent/'hopf_native_cost';sys.path.insert(0,str(COST))
from reference_compare import budget_samples

def main():
 dst=HERE/'individual_comparison.json';assert not dst.exists()
 protocol=json.loads((HERE/'protocol.json').read_text());path=COST/'panel_summary.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['files']['summary']['sha256']
 costs=json.loads(path.read_text());truth=json.loads((HERE/'reference_summary.json').read_text());assert truth['family_size']==25 and truth['nominal_joint_coverage']==.95
 boxes={(b['seed'],b['index']):b for b in truth['mc_truth_boxes']};assert len(boxes)==25
 rows=[]
 for row in costs['rows']:
  b=boxes[row['seed'],row['copy']];assert b['n']==100000000
  e=Q(row['value']);p=Q(b['hits'],b['n']);lo=Q(b['lower']);hi=Q(b['upper']);assert 0<=e<=1 and 0<lo<hi<1
  for name,per,ctl in [('dense_original',193,8),('analytic_zero_state_optimized',161,7)]:
   n,spent=budget_samples(row['conditional'],per,ctl)
   def ratio(p):return n*(e-p)**2/(p*(1-p))
   ends=[ratio(lo),ratio(hi)];lower=Q(0) if lo<=e<=hi else min(ends);upper=max(ends)
   rows.append(dict(seed=row['seed'],copy=row['copy'],sampler=name,candidate_source_budget=row['conditional'],sampling_draws=n,sampling_spent=spent,estimate=float(e),reference=float(p),point_ratio=float(ratio(p)),truth_box_ratio=[float(lower),float(upper)],exact_point_ratio=str(ratio(p)),exact_truth_box_ratio=[str(lower),str(upper)]))
 assert len(rows)==50 and len({(r['seed'],r['copy'],r['sampler']) for r in rows})==50
 out=dict(claim='426df9ce',rows=rows,truth_sha256=hashlib.sha256((HERE/'reference_summary.json').read_bytes()).hexdigest(),cost_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),truth_family=25,nominal_joint_coverage=.95,proof='For f(p)=(e-p)^2/[p(1-p)], derivative sign is sign(p-e), since numerator is (p-e)*(p+e-2*p*e) and second factor>=0 for e,p in[0,1]. Hence max at endpoints and min at e if inside, otherwise endpoint. Saved binary endpoints are treated exactly; beta inversion remains numerical.',scope='All25 individuals, no subset selection/new reference. Conditional source allowances and floating-forward/CP/native-runtime qualifications remain; upper allowance ratio above1 would not establish actual-cost failure.')
 dst.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':50,'max_point':max(r['point_ratio'] for r in rows),'max_upper':max(r['truth_box_ratio'][1] for r in rows),'upper_below_one':sum(r['truth_box_ratio'][1]<1 for r in rows)}))
if __name__=='__main__':main()

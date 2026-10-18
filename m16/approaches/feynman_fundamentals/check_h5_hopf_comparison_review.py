"""Exact saved single H5 budget/interval contraction, no new reference."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_native')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(P/'comparison.json');scope=read(P/'compare_scope.json');p=read(P/'pilot.json')
assert r['complete'] and p['complete'] and r['source_sha256']==sha(P/'compare.py')
for path,h in scope['pins'].items():assert sha(Path(path))==h
a=read(scope['truth_box_file']);assert a['family_size']==r['truth_box_family_size']==75
b=next(z for z in a['mc_truth_boxes'] if (z['H'],z['T'],z['seed'],z['index'])==(5,4,0,0));assert b==r['truth_box']
assert r['nominal_joint_coverage']==a['nominal_joint_coverage'] and r['numerical_beta_not_outward_rounded']==a['numerical_beta_not_outward_rounded']
e=F(p['value']);truth=F(b['hits'],b['n']);lo,hi=F(b['lower']),F(b['upper']);assert 0<=e<=1 and 0<lo<truth<hi<1
cost=p['conditional_source_with_constant_setup'];outrows=[]
for row,(name,per,batch) in zip(r['rows'],[('dense_original',281,8),('analytic_initial_zero',231,7)]):
 assert row['sampler']==name and row['candidate_conditional_source_budget']==cost
 assert (row['per_sample_with_aggregation'],row['per_batch_units'])==(per,batch)
 n=row['samples'];C=lambda x:per*x+batch*((x+999999)//1000000)+2
 assert C(n)==row['spent']<=cost<C(n+1)
 R=lambda v:F(n)*(e-v)**2/(v*(1-v));ends=[R(lo),R(hi)];lower=F() if lo<=e<=hi else min(ends);upper=max(ends)
 assert F(row['point_ratio']['fraction'])==R(truth) and row['point_ratio']['float']==float(R(truth))
 assert [F(x['fraction']) for x in row['nominal_truth_box_ratio_range']]==[lower,upper]
 assert [x['float'] for x in row['nominal_truth_box_ratio_range']]==[float(lower),float(upper)]
 assert F(row['error']['fraction'])==e-truth
 outrows.append(dict(sampler=name,samples=n,point_ratio=float(R(truth)),box_lower=float(lower),box_upper=float(upper)))
out=dict(complete=True,comparison_sha256=sha(P/'comparison.json'),rows=outrows,new_reference_or_model_calls=0,
 scope='Both integer budgets and exact endpoint/zero ratio extrema verified against original nominal75-family numerical10M confidence box. Upper ratios exceed one, so this finite comparison is inconclusive; an upper-cost allowance is not actual-cost failure evidence. No H4 result or new100M truth transfer.')
Path(__file__).with_name('h5_hopf_comparison_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

"""Exact own-budget ratio extrema for every fixed individual reference box."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,itertools,json
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_reference_precision')
C=P.parent/'hopf_native_cost'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=read(P/'individual_comparison.json');truth=read(P/'reference_summary.json');cost=read(C/'panel_summary.json')
assert d['truth_sha256']==sha(P/'reference_summary.json') and d['cost_sha256']==sha(C/'panel_summary.json')
assert d['truth_family']==25 and d['nominal_joint_coverage']==.95
models={(r['seed'],r['copy']):r for r in cost['rows']}
boxes={(r['seed'],r['index']):r for r in truth['mc_truth_boxes']}
assert len(d['rows'])==50
assert {(r['seed'],r['copy'],r['sampler']) for r in d['rows']}==set(itertools.product(range(5),range(5),('dense_original','analytic_zero_state_optimized')))
for r in d['rows']:
    model=models[r['seed'],r['copy']];b=boxes[r['seed'],r['copy']]
    e,p=F(model['value']),F(b['hits'],b['n']);lo,hi=F(b['lower']),F(b['upper'])
    assert 0<=e<=1 and 0<lo<hi<1 and b['n']==100000000
    per,ctl=(193,8) if r['sampler']=='dense_original' else (161,7)
    budget=model['conditional'];n=r['sampling_draws']
    spent=lambda n:per*n+ctl*((n+999999)//1000000)+2
    assert r['candidate_source_budget']==budget and spent(n)==r['sampling_spent']<=budget<spent(n+1)
    f=lambda x:n*(e-x)**2/(x*(1-x))
    ends=(f(lo),f(hi));lower=F(0) if lo<=e<=hi else min(ends);upper=max(ends)
    assert F(r['exact_point_ratio'])==f(p) and list(map(F,r['exact_truth_box_ratio']))==[lower,upper]
    assert r['point_ratio']==float(f(p)) and r['truth_box_ratio']==[float(lower),float(upper)]
worst=max(d['rows'],key=lambda r:F(r['exact_truth_box_ratio'][1]))
out=dict(complete=True,individual_comparisons=50,all_upper_below_one=all(F(r['exact_truth_box_ratio'][1])<1 for r in d['rows']),
         worst={k:worst[k] for k in ('seed','copy','sampler','point_ratio','truth_box_ratio','sampling_draws','candidate_source_budget')},
         input_sha256=sha(P/'individual_comparison.json'),new_model_reference_calls=0,
         scope='Exact ratio extrema for all 25 fixed individuals and both samplers using the same nominal simultaneous 25-truth boxes. Numerical confidence inversion, floating reference and mixed source allowance limitations persist; no full-runtime or universal claim.')
(HERE/'hopf_individual_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

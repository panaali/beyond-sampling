"""Exact independent integer budgets and saved truth-box contractions."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json,itertools
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_native_cost')
R=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_reference_precision')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(a):return sum(a,Q(0))/len(a)
def spent(n,p,c):return p*n+c*((n+999999)//1000000)+2
def draws(b,p,c):
    lo,hi=0,b//p+1
    while lo+1<hi:
        mid=(lo+hi)//2
        if spent(mid,p,c)<=b:lo=mid
        else:hi=mid
    assert spent(lo,p,c)<=b<spent(lo+1,p,c)
    return lo
a=read(R/'fresh_comparison.json');summary=read(P/'panel_summary.json');truth=read(R/'reference_summary.json')
assert a['summary_sha256']==sha(P/'panel_summary.json') and a['truth_audit_sha256']==sha(R/'reference_summary.json')
assert a['truth_family']==truth['family_size']==25 and a['truth_nominal_joint_coverage']==.95
assert not a['noise_subtraction'] and not a['h128_truth']
models={(r['seed'],r['copy']):r for r in summary['rows']}
boxes={(r['seed'],r['index']):r for r in truth['mc_truth_boxes'] if r['H']==4}
assert len(a['groups'])==60
keys={(g['seed'],g['population'],g['sampler'],g['allocation']) for g in a['groups']}
assert keys==set(itertools.product(range(5),('clean','noise','pooled'),('dense_original','analytic_zero_state_optimized'),('uniform','matched')))
pooled=[]
for g in a['groups']:
    inds={'clean':[0],'noise':[1,2,3,4],'pooled':[0,1,2,3,4]}[g['population']]
    assert inds==g['copies']
    rows=[models[g['seed'],i] for i in inds];bb=[boxes[g['seed'],i] for i in inds]
    costs=[r['conditional'] for r in rows];assert costs==g['candidate_source_budgets']
    budgets=costs if g['allocation']=='matched' else [sum(costs)//len(costs)]*len(costs)
    per,ctl=(193,8) if g['sampler']=='dense_original' else (161,7)
    nn=[draws(b,per,ctl) for b in budgets]
    assert nn==g['sampling_draws'] and [spent(n,per,ctl) for n in nn]==g['sampling_spent']
    errors=[];vars=[];emin=[];emax=[];vmin=[];vmax=[]
    for r,b,n in zip(rows,bb,nn):
        e=Q(r['value']);p=Q(b['hits'],b['n']);lo,hi=Q(b['lower']),Q(b['upper'])
        errors.append((e-p)**2);vars.append(p*(1-p)/n)
        emin.append(0 if lo<=e<=hi else min((e-lo)**2,(e-hi)**2))
        emax.append(max((e-lo)**2,(e-hi)**2))
        vmin.append(min(lo*(1-lo),hi*(1-hi))/n)
        vmax.append((Q(1,4) if lo<=Q(1,2)<=hi else max(lo*(1-lo),hi*(1-hi)))/n)
    ratio=mean(errors)/mean(vars);lower=mean(emin)/mean(vmax);upper=mean(emax)/mean(vmin)
    assert ratio==Q(g['exact_raw_point_ratio']) and [lower,upper]==list(map(Q,g['exact_box_ratio']))
    assert float(ratio)==g['raw_point_ratio'] and [float(lower),float(upper)]==g['simultaneous_truth_box_ratio']
    assert float(mean(errors))==g['raw_mse'] and float(mean(vars))==g['sampling_variance']
    if g['population']=='pooled' and g['allocation']=='matched' and g['sampler']=='analytic_zero_state_optimized':
        pooled.append(dict(seed=g['seed'],point=float(ratio),lower=float(lower),upper=float(upper)))
out=dict(complete=True,groups=60,max_point_ratio=max(g['raw_point_ratio'] for g in a['groups']),
         optimized_matched_pooled=pooled,max_box_upper=max(g['simultaneous_truth_box_ratio'][1] for g in a['groups']),all_upper_below_one=all(g['simultaneous_truth_box_ratio'][1]<1 for g in a['groups']),all_box_lower_zero=all(g['simultaneous_truth_box_ratio'][0]==0 for g in a['groups']),
         scope='Exact saved-endpoint contractions and maximal integer sample budgets. Fresh fixed25-family numerical CP inversion is not outward certified; mixed source allowances are not complete runtime cost. All-upper-below-one statement is conditional on the numerical confidence boxes and source allowances, not full runtime or universal accuracy.',new_model_reference_calls=0)
(HERE/'hopf_fresh_comparison_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

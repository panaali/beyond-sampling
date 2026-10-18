"""Independent exact budgets and all H5 confidence-box contractions."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_reference');NATIVE=P.parent/'hopf_h5_native';H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(encoded,value):assert F(encoded['fraction'])==value and encoded['float']==float(value)
def budget(C,per,batch):
 cost=lambda n:per*n+batch*((n+999999)//1000000)+2
 lo,hi=0,C//per+1
 while hi-lo>1:
  mid=(lo+hi)//2
  if cost(mid)<=C:lo=mid
  else:hi=mid
 assert lo>0 and cost(lo)<=C<cost(lo+1);return lo,cost(lo)
def ratio(n,e,p):return n*(e-p)**2/(p*(1-p))
def avg(xs):return sum(xs,F())/len(xs)
assert not (H/'h5_fresh_panel_comparison_review_results.json').exists()
start=time.process_time();r=read(P/'comparison.json');s=read(NATIVE/'panel_summary.json');review=read(H/'h5_hopf_global_review_results.json');p=read(P/'comparison_scope.json');fresh=read(H/'h5_reference_complete_review_results.json')
assert r['complete'] and s['complete'] and review['complete'] and review['summary_sha256']==sha(NATIVE/'panel_summary.json')==r['summary_sha256']
for path,h in p['pins'].items():assert sha(path)==h
assert r['source_sha256']==sha(P/'compare_panel.py') and r['truth_box_file_sha256']==sha(p['truth_box_file'])
audit=read(p['truth_box_file']);boxes={(b['seed'],b['copy']):b for b in audit['rows']};rows={(x['seed'],x['copy']):x for x in s['rows']};models={(x['seed'],x['copy']):x for x in read(NATIVE/'models.json')['models']}
assert set(rows)==set(boxes)==set(models)=={(s,c) for s in range(5) for c in range(5)}
assert r['family_size']==audit['family_size']==25 and r['nominal_joint_coverage']==audit['nominal_joint_coverage']==.95 and r['numerical_beta_not_outward_rounded'] and not r['noise_subtraction'] and not r['h128_truth']
assert fresh['pass'] and fresh['summary_sha256']==sha(p['truth_box_file']) and audit['draws']==2500000000 and audit['blocks']==2500
for key,b in boxes.items():assert b['n']==100000000 and type(b['hits']) is int and rows[key]['model_sha256']==models[key]['model_sha256']==b['model_sha256']
conventions={'dense_original':(281,8),'analytic_initial_zero':(231,7)};seen=set();individual_pass=individual_inconclusive=0;individual_max=F();individual_pointmax=F()
for g in r['individuals']:
 key=(g['seed'],g['copy']);name=g['sampler'];assert (key,name) not in seen;seen.add((key,name))
 e=F(rows[key]['value']);b=boxes[key];p0=F(b['hits'],b['n']);lo,hi=F(b['lower']),F(b['upper']);C=rows[key]['conditional'];n,cost=budget(C,*conventions[name]);assert (C,n,cost)==(g['candidate_budget'],g['draws'],g['spent'])
 point=ratio(n,e,p0);ends=[ratio(n,e,q) for q in (lo,hi)];lower=F() if lo<=e<=hi else min(ends);upper=max(ends)
 check(g['point_ratio'],point);check(g['nominal_truth_box_ratio_range'][0],lower);check(g['nominal_truth_box_ratio_range'][1],upper)
 individual_pass+=upper<1;individual_inconclusive+=lower<=1<=upper;individual_max=max(individual_max,upper);individual_pointmax=max(individual_pointmax,point)
assert len(seen)==len(r['individuals'])==50
seen=set();group_pass=group_inconclusive=0;group_max=group_pointmax=F();group_rows=[]
for g in r['groups']:
 seed,pop,name,alloc=g['seed'],g['population'],g['sampler'],g['allocation'];key=seed,pop,name,alloc;assert key not in seen;seen.add(key)
 copies={'clean':[0],'noise':[1,2,3,4],'pooled':[0,1,2,3,4]}[pop];assert g['copies']==copies
 costs=[rows[seed,c]['conditional'] for c in copies];assert costs==g['candidate_budgets'];budgets=costs if alloc=='matched' else [sum(costs)//len(costs)]*len(costs);assert alloc in ('matched','uniform')
 ns,spent=map(list,zip(*(budget(C,*conventions[name]) for C in budgets)));assert ns==g['draws'] and spent==g['spent'] and sum(spent)<=sum(costs)
 errors=[];variances=[];errlo=[];errhi=[];varlo=[];varhi=[]
 for c,n in zip(copies,ns):
  e=F(rows[seed,c]['value']);b=boxes[seed,c];p0=F(b['hits'],b['n']);lo,hi=F(b['lower']),F(b['upper']);errors.append((e-p0)**2);variances.append(p0*(1-p0)/n)
  errlo.append(F() if lo<=e<=hi else min((e-lo)**2,(e-hi)**2));errhi.append(max((e-lo)**2,(e-hi)**2));varlo.append(min(lo*(1-lo),hi*(1-hi))/n);varhi.append((F(1,4) if lo<=F(1,2)<=hi else max(lo*(1-lo),hi*(1-hi)))/n)
 mse=avg(errors);var=avg(variances);point=mse/var;lower=avg(errlo)/avg(varhi);upper=avg(errhi)/avg(varlo)
 for field,value in [('raw_mse',mse),('sampling_variance',var),('point_ratio',point)]:check(g[field],value)
 check(g['nominal_truth_box_ratio_range'][0],lower);check(g['nominal_truth_box_ratio_range'][1],upper)
 group_pass+=upper<1;group_inconclusive+=lower<=1<=upper;group_max=max(group_max,upper);group_pointmax=max(group_pointmax,point)
 group_rows.append(dict(seed=seed,population=pop,sampler=name,allocation=alloc,point=float(point),lower=float(lower),upper=float(upper)))
assert seen=={(seed,pop,name,alloc) for seed in range(5) for pop in ('clean','noise','pooled') for name in conventions for alloc in ('uniform','matched')}
out=dict(complete=True,groups=60,individuals=50,groups_upper_below_one=group_pass,groups_inconclusive=group_inconclusive,individuals_upper_below_one=individual_pass,individuals_inconclusive=individual_inconclusive,max_group_point=float(group_pointmax),max_group_upper=float(group_max),max_individual_point=float(individual_pointmax),max_individual_upper=float(individual_max),group_rows=group_rows,comparison_sha256=sha(P/'comparison.json'),review_cpu=time.process_time()-start,new_evaluator_reference_or_MC_calls=0,scope='All original fixed H5 saved comparisons, exact integer own/uniform budgets and fresh nominal25-family interval contractions independently verified. Numerical beta endpoints are not outward certificates. Conditional source allowance is not complete native/runtime cost; allowance ratio above1 alone is not actual-cost failure; no corrected-mask/general-rung claim.')
(H/'h5_fresh_panel_comparison_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='group_rows'}))

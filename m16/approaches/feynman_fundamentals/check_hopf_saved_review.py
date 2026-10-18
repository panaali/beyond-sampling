import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,math,time
from scipy.stats import binom
from scipy.optimize import brentq
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review')
def read(p):return json.loads(p.read_text())
def near(a,b,tol=3e-12):assert abs(a-b)<=tol*max(1,abs(a),abs(b)),(a,b)
def mean(xs):return sum(xs,F(0))/len(xs)
start=time.process_time();manifest=read(P/'manifest.json')
for name,row in manifest.items():assert hashlib.sha256((P/'frozen'/name).read_bytes()).hexdigest()==row['sha256']
roots=read(P/'root_controls.json');wrong=witness=0
for r,rescue in zip(roots['original'],roots['tile_rescue']):
 expected=sorted(x for a in r['a'] if a for x in [F(1,2),F(3,2)] if F(r['interval_over_pi'][0])<x<F(r['interval_over_pi'][1]))
 assert [float(x) for x in expected]==r['expected_over_pi']==rescue['actual_over_pi'];wrong+=not r['root_multiset_matches']
 if r['classification_witness']:
  w=r['classification_witness'];l,u=w['source_interval_over_pi'];a,b=w['first_angle_over_pi'],w['second_angle_over_pi'];assert l<a<b<u and math.cos(math.pi*a)*math.cos(math.pi*b)<0;witness+=1
assert (wrong,witness)==(4,3)
data=read(P/'frozen/ev_44.json');audit=read(P/'raw_audit.json');old=read(P/'frozen/run87.json');N=10000000
boxes={}
for b in audit['mc_truth_boxes']:
 row=data[str(b['seed'])]['rows'][b['index']];k=round(row['mc']*N);assert k==b['hits'];p=k/N
 l=brentq(lambda q:binom.sf(k-1,N,q)-.001,p-.01,p,xtol=5e-15);u=brentq(lambda q:binom.cdf(k,N,q)-.001,p,p+.01,xtol=5e-15)
 near(l,b['lower'],3e-14);near(u,b['upper'],3e-14);boxes[b['seed'],b['index']]=(b['lower'],b['upper'])
 prior=old['clean'][str(b['seed'])] if b['index']==0 else old['noisy'][str(b['seed'])]['rows'][b['index']-1]
 assert row['mc']==prior['mc1e7' if b['index']==0 else 'truth'] and row['recipe']==prior['est'] and row['recipe_ops']==prior['ops']
for g in audit['groups']:
 ids=list(range(5)) if g['population']=='pooled' else ([0] if g['population']=='clean' else list(range(1,5)))
 seed=g['seed'];key=g['estimator'];rows=[data[str(seed)]['rows'][i] for i in ids];p=[F(r['mc']) for r in rows];e=[F(r[key]) for r in rows];pq=mean([x*(1-x) for x in p]);mse=mean([(x-y)**2 for x,y in zip(p,e)]);ops=mean([F(r[key+'_ops']) for r in rows]);factor=ops/(96*pq)
 s=g['statistics'];near(float(mse*factor),s['ratio_mc_raw']);near(float(max(F(0),mse-pq/N)*factor),s['ratio_mc']);near(float((mse-pq/(N-1))),s['unbiased_marginal_mse_adjustment']);near(float(ops),s['ops'])
 lower_error=upper_error=lower_var=upper_var=0.
 for i,r in zip(ids,rows):
  l,u=boxes[seed,i];v=r[key];lower_error+=0 if l<=v<=u else min((v-l)**2,(v-u)**2);upper_error+=max((v-l)**2,(v-u)**2);lower_var+=min(l*(1-l),u*(1-u));upper_var+=.25 if l<=.5<=u else max(l*(1-l),u*(1-u))
 near(lower_error*float(ops)/(96*upper_var),g['simultaneous_truth_box']['ratio_lower']);near(upper_error*float(ops)/(96*lower_var),g['simultaneous_truth_box']['ratio_upper'])
continuity=read(P/'continuity_control.json')
for row in continuity['rows']:
 r=row['r'];expected=.5 if r<1 else (3*math.pi/4-math.atan(r))/(2*math.pi);near(expected,row['analytic_fraction']);near(expected,row['source_fraction'])
fullcircle=read(P/'full_circle_control.json');analytic=(math.atan(1)-math.atan(.5))/(2*math.pi);near(analytic,fullcircle['analytic_fraction']);near(analytic,fullcircle['tile_rescue_fraction']);assert fullcircle['original_fraction']==.125
upper=[g['simultaneous_truth_box']['ratio_upper'] for g in audit['groups'] if g['estimator']=='h64' and g['population']=='pooled']
out=dict(continuity_controls=10,full_circle_original=.125,full_circle_analytic=analytic,root_controls=8,original_root_multiset_failures=wrong,original_sign_change_witnesses=witness,local_tile_controls=8,binomial_truth_boxes=25,statistic_groups=30,h64_pooled_upper_bounds=upper,review_cpu=time.process_time()-start,scope='Pinned-source and saved-record audit, no model inputs or estimator run. Bonferroni nominal95% requires each saved MC truth to be binomial; numerical inversion is not outward-rounded.96-operation denominator and native counts are not a full cost certificate. Root failure extends to one explicit synthetic full circle, not a saved trained-case failure.')
Path(__file__).with_name('hopf_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

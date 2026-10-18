import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time
from scipy.stats import binom
from scipy.optimize import brentq
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review');start=time.process_time()
def read(p):return json.loads(p.read_text())
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=4e-12,abs_tol=5e-14),(a,b)
def mean(xs):return sum(xs,F(0))/len(xs)
for name,row in read(P/'wider_manifest.json').items():assert hashlib.sha256((P/'wider_frozen'/name).read_bytes()).hexdigest()==row['sha256']
d=read(P/'wider_audit.json');data={h:read(P/'wider_frozen'/f'ev_{h}4.json') for h in [4,5,6]}
for h in data:
 old=read(P/'frozen'/f'ev_{h}4.json');assert all(data[h][s]==v for s,v in old.items())
boxes={};N=10000000;tail=.05/150
for b in d['mc_truth_boxes']:
 p=data[b['H']][str(b['seed'])]['rows'][b['index']]['mc'];k=round(p*N);assert k==b['hits'] and b['n']==N
 lo=brentq(lambda q:binom.sf(k-1,N,q)-tail,0,p,xtol=5e-15);hi=brentq(lambda q:binom.cdf(k,N,q)-tail,p,1,xtol=5e-15);near(lo,b['lower']);near(hi,b['upper']);boxes[b['H'],b['seed'],b['index']]=(b['lower'],b['upper'])
cleanloss=[];upperpass=[]
for g in d['groups']:
 h=g['H'];seed=g['seed'];ids=range(5) if g['population']=='pooled' else ([0] if g['population']=='clean' else range(1,5));rows=[data[h][str(seed)]['rows'][i] for i in ids];sample=4*h*(h+2);p=[F(r['mc']) for r in rows];est=[F(r['h64']) for r in rows];op=[F(r['h64_ops']) for r in rows];pq=[x*(1-x) for x in p];mse=mean([(x-y)**2 for x,y in zip(p,est)]);uniform=sample*mean(pq)/mean(op);matched=mean([sample*v/o for v,o in zip(pq,op)])
 near(mse/uniform,g['raw_uniform_ratio']);near(mse/matched,g['raw_matched_ratio']);near(max(F(0),mse-mean(pq)/N)/uniform,g['adjusted_uniform_ratio']);near(mean([(F(r['h64'])-F(r['ref']))**2 for r in rows])/uniform,g['finite_reference_uniform_ratio'])
 elo=[];ehi=[];vlo=[];vhi=[]
 for i,r in zip(ids,rows):
  l,u=boxes[h,seed,i];e=r['h64'];elo.append(0 if l<=e<=u else min((e-l)**2,(e-u)**2));ehi.append(max((e-l)**2,(e-u)**2));vlo.append(min(l*(1-l),u*(1-u)));vhi.append(.25 if l<=.5<=u else max(l*(1-l),u*(1-u)))
 ub=[sum(elo)*float(mean(op))/(sample*sum(vhi)),sum(ehi)*float(mean(op))/(sample*sum(vlo))];mb=[sum(elo)/sum(sample*v/float(o) for v,o in zip(vhi,op)),sum(ehi)/sum(sample*v/float(o) for v,o in zip(vlo,op))]
 for a,b in zip(ub,g['box_uniform']):near(a,b)
 for a,b in zip(mb,g['box_matched']):near(a,b)
 if g['population']=='clean' and g['raw_uniform_ratio']>1:cleanloss.append(dict(H=h,seed=seed,point=g['raw_uniform_ratio'],box=g['box_uniform']))
 if g['box_uniform'][1]<1:upperpass.append((h,seed,g['population']))
assert len(boxes)==75 and len(d['groups'])==45 and len(cleanloss)==3
out=dict(cases=75,groups=45,previous_rows_unchanged=True,clean_point_ratios_above_one=cleanloss,groups_with_box_upper_below_one=upperpass,review_cpu=time.process_time()-start,scope='Saved arithmetic and independent binomial-tail inversion.75-case Bonferroni coverage conditional on binomial MC provenance; numerical endpoints not outward. Native96/140/192 costs unaudited, finite-grid reference not exact truth; no rung acceptance or trained-source replay.')
Path(__file__).with_name('hopf_wider_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

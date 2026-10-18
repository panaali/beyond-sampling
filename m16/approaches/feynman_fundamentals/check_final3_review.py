"""Independent fixed-panel arithmetic and binomial interval tail check."""
from pathlib import Path
from fractions import Fraction as F
import json,math,time
from scipy.special import betainc
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_78/m16/approaches/final3_review');audit=json.loads((src/'audit_results.json').read_text());raw=json.loads((src/'snapshot/res_H8_T4.json').read_text());exact=json.loads((src/'snapshot/exact_H8_T4.json').read_text());M=4000000;alpha=.05/len(raw);max_tail=0.;ratios=[]
for r in audit['rows']:
 v=raw[r['name']];p=F(str(v['truth']));e=F(str(v['est']));n=F(v['ops'])*3/2000;variance=p*(1-p);ratio=n*(e-p)**2/variance;corrected=n*max((e-p)**2-variance/M,0)/variance
 assert abs(float(ratio)-r['raw_ratio'])<1e-10 and abs(float(corrected)-r['corrected_ratio'])<1e-10
 k=int(p*M);assert p*M==k==r['truth_successes'];lo,hi=r['truth_simultaneous95_interval'];res=max(abs(betainc(k,M-k+1,lo)-alpha/2),abs(betainc(k+1,M-k,hi)-(1-alpha/2)));max_tail=max(max_tail,res);assert res<2e-11
 # For e in [0,1], the ratio is decreasing before e and increasing after e.
 def f(q):return float(n)*(float(e)-q)**2/(q*(1-q))
 lower=0 if lo<=float(e)<=hi else min(f(lo),f(hi));upper=max(f(lo),f(hi));assert abs(lower-r['ratio_over_interval'][0])<1e-10 and abs(upper-r['ratio_over_interval'][1])<1e-10;ratios.append(lower)
 if r['name'] in exact:
  gap=n*(e-F(str(exact[r['name']]['acc'])))**2/variance;assert abs(float(gap)-r['gap_ratio'])<1e-10
zoo=json.loads((src/'snapshot/evalH_8.json').read_text())['zoo'];mean=lambda x:math.fsum(x)/len(x)
c3=mean([v['clean_cells'] for v in zoo.values()]);r3=mean([v['regions'] for v in zoo.values()]);assert c3==466 and r3==235.6
assert abs(c3/r3-audit['trends']['T3_exact_ratio'])<1e-14
H,T,r=16,10,16;prefix=math.factorial(T)*(4*H+2*H*r);margin=4*math.factorial(T)*T*H*(T+1);assert prefix==2090188800 and margin==25546752000
out=dict(models=len(raw),minimum_simultaneous_ratio=min(ratios),maximum_beta_tail_residual=max_tail,T3_exact_cell_region_ratio=c3/r3,unchanged_completion_prefix=prefix,unchanged_recipe108_first_margin_bound=margin,review_cpu=time.process_time()-start,scope='Fixed9 models and declared nominal counter; no full hardware cost, model-population claim, or impossibility for redesigned charts.')
(here/'final3_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

"""Direct Gaussian line-density integration, no author CDF formula."""
from pathlib import Path
from fractions import Fraction as F
import json,time,resource
import mpmath as mp
resource.setrlimit(resource.RLIMIT_CPU,(20,20));here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_78/m16/approaches/gaussian_margin_slice');c=json.loads((src/'certificate.json').read_text())['certificate'];r=json.loads((src/'results.json').read_text());signs=json.loads((src/'sign_results.json').read_text());start=time.process_time();x=list(map(F,c['pair'][0]['x']));y=list(map(F,c['pair'][1]['x']));d=[b-a for a,b in zip(x,y)];D=sum(v*v for v in d);mu=-sum(a*b for a,b in zip(x,d))/D;var=1/D;cut=F(r['crossing']);assert mu==F(r['mu']) and var==F(r['variance']);mp.mp.dps=70
num=lambda q:mp.mpf(q.numerator)/q.denominator
xx=list(map(num,x));dd=list(map(num,d));cc=num(cut)
def density(t):return mp.exp(-sum((a+t*b)**2 for a,b in zip(xx,dd))/2)
rows=[]
for index,(a,b) in enumerate(((F(0),F(1)),(F(1,2),F(1)))):
 aa,bb=num(a),num(b);Z=mp.quad(density,[aa,cc,bb]);P=mp.quad(density,[aa,cc])/Z;mean=mp.quad(lambda t:t*density(t),[aa,cc,bb])/Z;old=r['evaluations'][-1]['rows'][index];delta=max(abs(P-mp.mpf(old['correct_probability'])),abs(mean-mp.mpf(old['truncated_mean'])));assert delta<mp.mpf('1e-60')
 q=lambda t:(t-mu)**2/(2*var);spread=max(q(a),q(b))-q(max(a,min(b,mu)));assert spread==F(signs['rows'][index]['exponent_spread'])<F(1,2);assert 1/(1-spread)<2
 left=(cut-a)**2;right=(b-cut)**2;bound=2*right-left if index==0 else right-2*left;assert bound==F(signs['rows'][index]['twice_normalized_moment_bound']);assert bound<0 if index==0 else bound>0
 plugin=int(mean<=cc);rows.append(dict(interval=[str(a),str(b)],direct_correct_probability=float(P),direct_truncated_mean=float(mean),plugin_bias=float(plugin-P),maximum_independent_difference=mp.nstr(delta,8),exact_mean_sign=-1 if index==0 else 1))
out=dict(rows=rows,review_cpu=time.process_time()-start,scope='Independent direct unnormalized Gaussian integrals plus exact rational sign inequalities. Conditional slices only, not E27 actual omitted-variable distribution or global accuracy.')
(here/'gaussian_slice_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

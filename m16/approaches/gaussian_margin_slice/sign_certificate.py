"""Exact rational certification of both local conditional-mean bias signs."""
from pathlib import Path
from fractions import Fraction as F
import json,time
HERE=Path(__file__).parent
start=time.process_time();r=json.loads((HERE/'results.json').read_text());mu=F(r['mu']);v=F(r['variance']);c=F(r['crossing']);rows=[]
for a,b,expected in ((F(0),F(1),-1),(F(1,2),F(1),1)):
    assert a<c<b and v>0
    exponent=lambda t:(t-mu)**2/(2*v)
    minimum=exponent(max(a,min(b,mu)));maximum=max(exponent(a),exponent(b));spread=maximum-minimum;assert 0<=spread<F(1,2)
    # Series comparison exp(t)<=sum(t**n)=1/(1-t), 0<=t<1.
    rational_density_ratio_upper=1/(1-spread);assert rational_density_ratio_upper<2
    pos=(b-c)**2;neg=(c-a)**2
    moment_bound=2*pos-neg if expected<0 else pos-2*neg
    assert expected*moment_bound>0
    rows.append(dict(interval=[str(a),str(b)],exponent_spread=str(spread),exponent_spread_float=float(spread),density_ratio_rational_upper=str(rational_density_ratio_upper),density_ratio_upper_float=float(rational_density_ratio_upper),twice_normalized_moment_bound=str(moment_bound),moment_bound_float=float(moment_bound),mean_minus_crossing_sign=expected,plugin_correct=int(expected<0),bias_sign=-expected))
out=dict(rows=rows,cpu_seconds=time.process_time()-start,scope='Exact rational exponent and moment inequalities plus exp(t)<=1/(1-t) prove signs only; numerical conditional probability magnitudes remain non-enclosed MP values.')
(HERE/'sign_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(rows=[{k:q[k] for k in ('interval','exponent_spread_float','density_ratio_upper_float','moment_bound_float','bias_sign')} for q in rows],cpu_seconds=out['cpu_seconds'])))

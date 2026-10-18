"""Conditional finite and generic helper upper contractions; no native rerun."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')

def active(rows):
    c=Counter()
    for r in rows:
        if r['kernel']=='dlarfg' and r['m']>1:
            assert r['n']>0 and not r['query'] and r['failures']==0
            c[r['m'],r['n']]+=r['calls']
    return c

def price(c):
    n=sum(c.values());length=sum((m-1)*v for (m,s),v in c.items())
    return dict(active_reflectors=n,scale_length_sum=length,
      conditional_no_rescale=dict(dlapy2_upper=13*n,dscal_upper=length+3*n,direct_dlamch_calls_upper=2*n,internal_dlamch_calls_upper=n,helper_named_numeric_upper=16*n+length),
      generic_up_to20_rescales=dict(dlapy2_upper=26*n,dscal_upper=21*(length+3*n),direct_dlamch_calls_upper=2*n,internal_dlamch_calls_upper=2*n,helper_named_numeric_upper=89*n+21*length))

def main():
    assert not (HERE/'contraction.json').exists()
    start=time.process_time();s=json.loads((CENSUS/'summary.json').read_text());out={};pins={}
    norm=json.loads((HERE.parent/'norm_review/contraction.json').read_text())
    for mode in ['exact','recipe108']:
        geo=Counter();cold=None
        for j in range(8):
            p=CENSUS/f'{mode}_{j}.json';pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text())
            geo+=active(d['native']['kernels']);c=active(d['cold']['kernels'])
            if cold is None:cold=c
            else:assert cold==c
        assert geo+cold==active(s['modes'][mode]['rows'])
        phases={k:price(v) for k,v in dict(geometry=geo,cold_once=cold,total=geo+cold).items()}
        r=phases['total'];a=r['active_reflectors'];norm_units=norm['modes'][mode]['total']['named_numeric_subset']
        assert a==norm['modes'][mode]['total']['calls']
        phases['composed_named_subset']=dict(norm_observed=norm_units,
          conditional_reflector_exclusive=10*a,
          conditional_helpers=r['conditional_no_rescale']['helper_named_numeric_upper'],
          conditional_total_upper=norm_units+10*a+r['conditional_no_rescale']['helper_named_numeric_upper'],
          generic_reflector_exclusive=113*a,
          generic_helpers=r['generic_up_to20_rescales']['helper_named_numeric_upper'],
          generic_total_upper=norm_units+113*a+r['generic_up_to20_rescales']['helper_named_numeric_upper'])
        out[mode]=phases
    result=dict(modes=out,pins=pins,cpu=time.process_time()-start,
      scope='Finite installed named numeric subsets, not complete rank/runtime cost. Conditional no-rescale composition needs complete historical observer/binding provenance. Generic source upper retains up to20 rescales and2hypot/21scale calls per active reflector, while observed global norms are charged once. N<=1 returns before floating work. DLAMCH character control unpriced, not free runtime. No full parent prices added.')
    (HERE/'contraction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:v['composed_named_subset'] for m,v in out.items()}))
if __name__=='__main__':main()

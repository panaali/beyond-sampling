"""Finite Q3 parent-exclusive source upper, actual Q5 count not retry guess."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')

def hist(rows):
    c=Counter()
    for r in rows:
        if r['kernel'] in ['dlasq3','dlasq5','dlasq6']:
            assert not r['query'] and r['failures']==0
            c[r['kernel'],r['m'],r['n']]+=r['calls']
    return c

def price(c):
    base={1:3,2:25,3:51,4:73};q3=sum(v for (k,n,p),v in c.items() if k=='dlasq3');q5=sum(v for (k,n,p),v in c.items() if k=='dlasq5');q6=sum(v for (k,n,p),v in c.items() if k=='dlasq6');assert q6==0
    b=sum(base[n]*v for (k,n,p),v in c.items() if k=='dlasq3')
    return dict(q3_calls=q3,q5_actual_calls=q5,q6_actual_calls=q6,parent_base_upper=b,parent_retry_upper=16*q5,parent_source_upper=b+16*q5)

def main():
    assert not (HERE/'parent_contraction.json').exists()
    controls=json.loads((HERE/'parent_controls.json').read_text());retry=json.loads((HERE/'retry_controls.json').read_text());assert controls['failures']==0 and retry['failures']==0 and retry['max_q5']>1
    start=time.process_time();summary=json.loads((CENSUS/'summary.json').read_text());out={};pins={}
    for mode in ['exact','recipe108']:
        geo=Counter();cold=None
        for j in range(8):
            p=CENSUS/f'{mode}_{j}.json';pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text());assert d['native']['overflow']==0 and d['cold']['overflow']==0
            geo+=hist(d['native']['kernels']);c=hist(d['cold']['kernels'])
            if cold is None:cold=c
            else:assert cold==c
        assert geo+cold==hist(summary['modes'][mode]['rows'])
        out[mode]={k:price(v) for k,v in dict(geometry=geo,cold_once=cold,total=geo+cold).items()}
    result=dict(modes=out,pins=pins,cpu=time.process_time()-start,scope='Q3 parent-exclusive source upper, actual global Q5 counts safely upper-bound Q3 attempts; Q6 absent. Native children not priced here. Includes one cold setup, not eight diagnostic launches. Conditional source/binary provenance, finite parity only, integer/control/memory unpriced. Not full rank cost.')
    (HERE/'parent_contraction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:d['total'] for m,d in out.items()}))
if __name__=='__main__':main()

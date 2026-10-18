"""Q4 source upper from saved Q3 entries; no estimator/native rerun."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')

def hist(rows):
    c=Counter()
    for r in rows:
        if r['kernel']=='dlasq3':
            assert r['m'] in [1,2,3,4] and not r['query'] and r['failures']==0
            c[r['m'],r['n']]+=r['calls']
    return c

def price(c):
    bounds={1:0,2:0,3:35,4:44}
    return dict(q3_calls=sum(c.values()),q4_calls_upper=sum(v for (n,p),v in c.items() if n>=3),source_named_numeric_upper=sum(bounds[n]*v for (n,p),v in c.items()),rows=[dict(entry_length=n,pp=p,q3_calls=v,q4_source_unit_upper=bounds[n]) for (n,p),v in sorted(c.items())])

def main():
    assert not (HERE/'contraction.json').exists()
    start=time.process_time();s=json.loads((CENSUS/'summary.json').read_text());out={};pins={}
    for mode in ['exact','recipe108']:
        geo=Counter();cold=None
        for j in range(8):
            p=CENSUS/f'{mode}_{j}.json';pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text())
            assert d['native']['overflow']==0 and d['cold']['overflow']==0
            geo+=hist(d['native']['kernels']);c=hist(d['cold']['kernels'])
            if cold is None:cold=c
            else:assert cold==c
        assert geo+cold==hist(s['modes'][mode]['rows'])
        out[mode]={k:price(v) for k,v in dict(geometry=geo,cold_once=cold,total=geo+cold).items()}
    result=dict(modes=out,pins=pins,cpu=time.process_time()-start,scope='Pinned-source Q4 numeric upper using Q3 initial-length histogram, at most one Q4 per Q3 after deflation. Cold once; Q3/Q5 excluded. Finite compiled controls are not all-input equivalence proof. No geometry/native/rank rerun or complete-cost claim.')
    (HERE/'contraction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:d['total'] for m,d in out.items()}))
if __name__=='__main__':main()

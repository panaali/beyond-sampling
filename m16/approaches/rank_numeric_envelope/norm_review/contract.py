"""Contract finite original raw norm records, keeping cold work once."""
import collections
import hashlib
import json
from pathlib import Path
import time
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')

def counts(rows):
    out=collections.Counter()
    for r in rows:
        if r['kernel']!='dnrm2':continue
        assert not r['query'] and r['failures']==0 and r['m']>=1 and r['n']>0
        out[r['m'],r['n']]+=r['calls']
    return out

def price(counter):
    categories=collections.Counter(mul=0,add=0,sqrt=0,abs=0)
    for (n,stride),calls in counter.items():
        if n==1:categories['abs']+=calls
        else:
            categories['mul']+=n*calls
            categories['add']+=(n+3)*calls
            categories['sqrt']+=calls
    return dict(calls=sum(counter.values()),categories=dict(categories),named_numeric_subset=sum(categories.values()))

def main():
    assert not (HERE/'contraction.json').exists()
    start=time.process_time();summary=json.loads((CENSUS/'summary.json').read_text());pins={}
    out={}
    for mode in ['exact','recipe108']:
        geometry=collections.Counter();cold=None
        for j in range(8):
            p=CENSUS/f'{mode}_{j}.json';pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text())
            assert d['native']['overflow']==0 and d['cold']['overflow']==0
            geometry+=counts(d['native']['kernels']);c=counts(d['cold']['kernels'])
            if cold is None:cold=c
            else:assert c==cold
        total=geometry+cold
        assert total==counts(summary['modes'][mode]['rows'])
        assert cold==counts(summary['modes'][mode]['cold_setup_once']['kernels'])
        out[mode]=dict(geometry=price(geometry),cold_once=price(cold),total=price(total),rows=[dict(n=n,stride=s,calls=v,named_unit_cost=1 if n==1 else 2*n+4) for (n,s),v in sorted(total.items())])
    result=dict(modes=out,pins=pins,summary_sha256=hashlib.sha256((CENSUS/'summary.json').read_bytes()).hexdigest(),cpu=time.process_time()-start,scope='Installed DNRM2 named numeric subset, finite saved call records including cold once; historical binary/binding provenance remains a premise. Not complete runtime cost or a universal call bound. Parent exclusive counts exclude these child calls; do not add full parent prices or subtract abstract geometry norms without attribution.')
    (HERE/'contraction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{k:v for k,v in d.items() if k!='rows'} for m,d in out.items()}))
if __name__=='__main__':main()

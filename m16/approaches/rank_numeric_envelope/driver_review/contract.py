"""Remaining selected-driver source upper from saved GESDD shapes."""
from collections import Counter
from pathlib import Path
import hashlib,json,time
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')
def hist(phase):
    assert phase['overflow']==0;c=Counter()
    for r in phase['kernels']:
        if r['kernel']=='dgesdd':
            assert r['failures']==0
            c[r['m'],r['n'],r['query']]+=r['calls']
    return c
def price(c):
    rows=[]
    for (m,n,query),count in sorted(c.items()):
        k=min(m,n);assert 1<=k<=4 and max(m,n)<=24
        per=4 if query else 194+6*m*n+3*k
        rows.append(dict(m=m,n=n,query=query,calls=count,per_call_source_upper=per,source_upper=per*count))
    return dict(calls=sum(c.values()),source_upper=sum(r['source_upper'] for r in rows),rows=rows)
def main():
    out=HERE/'contraction.json';assert not out.exists()
    control=json.loads((HERE/'helper_controls.json').read_text());assert control['cases']==439 and control['failures']==0
    nb={(r['args']['name'],r['args']['m'],r['args']['n']) for r in control['rows'] if r['args']['kind']=='blocksize' and r['extra']['blocks_absent']}
    start=time.process_time();summary=json.loads((CENSUS/'summary.json').read_text());modes={};pins={}
    for mode in ['exact','recipe108']:
        geo=Counter();cold=None
        for i in range(8):
            p=CENSUS/f'{mode}_{i}.json';pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text());geo+=hist(d['native']);cc=hist(d['cold'])
            for phase in ['native','cold']:
                for r in d[phase]['kernels']:
                    if r['kernel'] in ['dgeqrf','dgelqf','dgebrd']:assert (r['kernel'],r['m'],r['n']) in nb
            if cold is None:cold=cc
            else:assert cold==cc
        assert geo+cold==hist(dict(kernels=summary['modes'][mode]['rows'],overflow=0))
        modes[mode]={k:price(v) for k,v in dict(geometry=geo,cold_once=cold,total=geo+cold).items()}
    result=dict(modes=modes,pins=pins,cpu_seconds=time.process_time()-start,scope='Selected successful finite values-only drivers, source-category upper including generic two scales and bidiagonal rotations/sorts. Existing unblocked/Q1 descendants excluded. One cold rank-geometry phase; larger cold eigensolver setup is not priced here. Integer/character/casts/control/memory/runtime and all-input compiled lane equivalence excluded. Not complete native/runtime cost.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:d['total']['source_upper'] for m,d in modes.items()}))
if __name__=='__main__':main()

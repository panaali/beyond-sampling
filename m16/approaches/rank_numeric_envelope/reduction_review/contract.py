"""Shape-based source upper for application scans; BLAS children unpriced."""
from collections import Counter
from pathlib import Path
import hashlib,json,time
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')

def applications(kernel,m,n):
    rows=[]
    for i in range(1,min(m,n)+1):
        if kernel=='dgeqr2' and i<n:rows.append(('L',m-i+1,n-i))
        elif kernel=='dgelq2' and i<m:rows.append(('R',n-i+1,m-i))
        elif kernel=='dgebd2':
            if m>=n and i<n:rows.extend([('L',m-i+1,n-i),('R',n-i,m-i)])
            elif m<n and i<m:rows.extend([('R',n-i+1,m-i),('L',m-i,n-i)])
    return rows

def hist(phase):
    assert phase['overflow']==0;c=Counter()
    for row in phase['kernels']:
        if row['kernel'] in ['dgeqr2','dgelq2','dgebd2']:
            assert not row['query'] and row['failures']==0
            c[row['kernel'],row['m'],row['n']]+=row['calls']
    return c

def price(c):
    shapes=Counter()
    for (k,m,n),count in c.items():
        for app in applications(k,m,n):shapes[app]+=count
    rows=[]
    for (s,l,o),count in sorted(shapes.items()):
        bound=(l+4+l*o) if s=='L' else (2*l+4+l*o)
        rows.append(dict(side=s,length=l,opposite=o,calls=count,per_call_source_upper=bound,source_upper=count*bound))
    return dict(application_calls=sum(shapes.values()),application_source_upper=sum(r['source_upper'] for r in rows),rows=rows)

def main():
    out=HERE/'contraction.json';assert not out.exists()
    control=json.loads((HERE/'controls.json').read_text());assert control['cases']==1152 and control['failures']==0
    start=time.process_time();summary=json.loads((CENSUS/'summary.json').read_text());modes={};pins={}
    for mode in ['exact','recipe108']:
        geo=Counter();cold=None
        for i in range(8):
            p=CENSUS/f'{mode}_{i}.json';pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text());geo+=hist(d['native']);c=hist(d['cold'])
            if cold is None:cold=c
            else:assert cold==c
        assert geo+cold==hist(dict(kernels=summary['modes'][mode]['rows'],overflow=0))
        modes[mode]={k:price(v) for k,v in dict(geometry=geo,cold_once=cold,total=geo+cold).items()}
    result=dict(modes=modes,pins=pins,cpu_seconds=time.process_time()-start,scope='Named DGEQR2/DGELQ2/DGEBD2 exclusive numeric source work is zero; this adds only their DLARF1F application scans/negations. Full dimensions upper-bound LASTV/LASTC shrinkage. Includes one cold setup, no DLARFG or BLAS child work. Does not price other reducers/wrappers, integer/control/memory/runtime. Finite ABI/source qualification; not complete rank cost.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:{k:v for k,v in d['total'].items() if k!='rows'} for m,d in modes.items()}))
if __name__=='__main__':main()

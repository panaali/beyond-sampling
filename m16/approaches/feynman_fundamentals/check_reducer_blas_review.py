"""Saved BLAS controls and nonoverlapping shape-ledger contraction only."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/reduction_review')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(P/'blas_controls.json');b=read(P/'blas_contraction.json');prior=read(P/'contraction.json')
assert a['source_sha256']==sha(P/'blas_controls.py') and a['cases']==len(a['rows'])==1344 and a['failures']==0
cases=Counter();keys=set()
for r in a['rows']:
    assert r['match'] and r['untouched'] and r['actual']==r['expected']
    cases[r['args']['kind']]+=1;k=json.dumps(r['args'],sort_keys=True);assert k not in keys;keys.add(k)
assert dict(cases)==dict(axpy=576,ger=432,gemv=336)
for name,h in b['pins'].items():assert sha(P/name)==h
totals={}
for mode,phases in b['modes'].items():
    for phase,s in phases.items():
        total=Counter()
        assert [x['shape'] for x in s['rows']]==prior['modes'][mode][phase]['rows']
        for x in s['rows']:
            r=x['shape'];L,C=r['length'],r['opposite'];side=r['side']
            assert L>=1 and C>=1
            if side=='L':assert L<=24 and C<=3
            else:assert side=='R' and L<=4 and C<=5
            if L==1:terms=dict(scal=C+3,gemv=0,axpy=0,ger=0)
            elif side=='L':terms=dict(scal=0,gemv=2*(L-1)*C+7*C+7,axpy=4*C+2,ger=2*(L-1)*C+C+1)
            else:terms=dict(scal=0,gemv=3*C*(L-1)+(L-1)+3*C+4,axpy=4*C+2,ger=2*C*(L-1)+(L-1)+1)
            terms['total']=sum(terms.values());assert terms==x['per_call_child_upper']
            if L>1:
                assert terms['total']==(4*L*C+8*C+10 if side=='L' else 5*C*L+2*C+2*L+5)
                assert terms['total']>=C+3
            for k,v in terms.items():total[k]+=v*r['calls']
        assert dict(total)==s['child_upper']
    for k in phases['total']['child_upper']:
        assert phases['total']['child_upper'][k]==phases['geometry']['child_upper'][k]+phases['cold_once']['child_upper'][k]
    totals[mode]=phases['total']['child_upper']
out=dict(complete=True,controls=dict(cases),totals=totals,controls_sha256=sha(P/'blas_controls.json'),
         contraction_sha256=sha(P/'blas_contraction.json'),new_native_calls=0,
         scope='Conditional small-dimension source/installed-kernel envelopes under historical binding and reached-path premises. Saved numerical controls do not count instructions. Excludes application scans, DLARFG, unrelated BLAS, drivers, control and memory; not full native cost.')
(HERE/'reducer_blas_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

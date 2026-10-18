"""Saved corrected reflector application controls and shape contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/reduction_review')
C=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(P/'controls.json');assert a['cases']==len(a['rows'])==1152 and a['failures']==0
for name,h in a['pins'].items():assert sha(P/name)==h
right_stress=0
for r in a['rows']:
    arg=r['args'];L,Cn=arg['length'],arg['other'];left=arg['side']=='L'
    expected=L+4+L*Cn if left else 2*L+4+L*Cn
    assert expected==r['upper'] and r['meta']['count']<=expected and r['match']
    assert r['expected']==r['actual'] and r['expected_work']==r['actual_work']
    assert r['input_v']==r['expected_v']==r['actual_v']
    right_stress+=not left and L>Cn and arg['apattern']==3
assert right_stress>0
s=read(P/'contraction.json');outm={}
for name,h in s['pins'].items():assert sha(C/name)==h
for mode in ('exact','recipe108'):
    groups={'geometry':Counter(),'cold_once':Counter()}
    for j in range(8):
        d=read(C/f'{mode}_{j}.json');cold=Counter()
        for phase in ('native','cold'):
            assert not d[phase]['overflow']
            dest=groups['geometry'] if phase=='native' else cold
            for r in d[phase]['kernels']:
                k,m,n=r['kernel'],r['m'],r['n']
                if k not in ('dgeqr2','dgelq2','dgebd2'):continue
                assert not r['query'] and not r['failures']
                for i in range(min(m,n)):
                    apps=[]
                    if k=='dgeqr2' and i+1<n:apps=[('L',m-i,n-i-1)]
                    if k=='dgelq2' and i+1<m:apps=[('R',n-i,m-i-1)]
                    if k=='dgebd2' and i+1<min(m,n):
                        apps=[('L',m-i,n-i-1),('R',n-i-1,m-i-1)] if m>=n else [('R',n-i,m-i-1),('L',m-i-1,n-i-1)]
                    for app in apps:dest[app]+=r['calls']
        if j==0:groups['cold_once']=cold
        else:assert cold==groups['cold_once']
    groups['total']=groups['geometry']+groups['cold_once']
    for phase,hist in groups.items():
        report=s['modes'][mode][phase]
        got={(r['side'],r['length'],r['opposite']):r for r in report['rows']}
        assert set(hist)==set(got)
        total=0
        for (side,L,Cn),count in hist.items():
            bound=L+4+L*Cn if side=='L' else 2*L+4+L*Cn
            rr=got[side,L,Cn];assert rr['calls']==count and rr['per_call_source_upper']==bound and rr['source_upper']==count*bound
            total+=count*bound
        assert total==report['application_source_upper'] and sum(hist.values())==report['application_calls']
    outm[mode]={k:v for k,v in s['modes'][mode]['total'].items() if k!='rows'}
out=dict(complete=True,controls=1152,right_allzero_L_greater_C_controls=right_stress,modes=outm,
         new_rank_native_calls=0,scope='Corrected side-specific source scan/negation upper only; shared native BLAS tests parent semantics, children remain unpriced. Three named reducer parents exclude index/copy work; no other reducer or full-cost inference.')
(HERE/'reduction_application_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

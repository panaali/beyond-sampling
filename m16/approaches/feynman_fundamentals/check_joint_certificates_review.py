"""Independent original-row, exact witness and direct-model T120 audit."""
import gzip,json,math,time,sys
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
start=time.process_time()
p=Path('/workspace/work/recover-small_h2_910/m16/approaches/bulk_joint_feasibility')/(sys.argv[1] if len(sys.argv)>1 else 'panel.json')
d=json.loads(p.read_text())
skip=0
if p.name=='panel_phase2.json':
    previous=json.loads(p.with_name('panel.json').read_text())['systems'];skip=len(previous)
    assert d['systems'][:skip]==previous
source=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz')
m=json.loads(gzip.decompress(source.read_bytes()))['model'];f=lambda v:F(int(v[0]),int(v[1]))
wi=list(map(f,m[0]));wh=[list(map(f,r)) for r in m[1]];wo=[list(map(f,r)) for r in m[2]]
def canonical(row):
    scale=math.lcm(*(v.denominator for v in row));z=[int(v*scale) for v in row];g=math.gcd(*z)
    return tuple(v//g for v in z) if g else tuple(z)
def maps(history):
    A=[[F(0)]*4 for _ in range(4)];gates=[]
    for t in range(4):
        B=[[sum(wh[i][k]*A[k][q] for k in range(4))+(wi[i] if q==t else 0) for q in range(4)] for i in range(4)]
        for i,r in enumerate(B):
            on=history[4*t+i]=='1'
            if any(r):gates.append(canonical(r if on else [-v for v in r]))
            else:assert not on
        A=[r if history[4*t+i]=='1' else [F(0)]*4 for i,r in enumerate(B)]
    L=[[sum(wo[i][k]*A[k][q] for k in range(4)) for q in range(4)] for i in range(4)]
    return list(dict.fromkeys(gates)),L
cache={};counts=Counter();forward_checks=0
for s in d['systems'][skip:]:
    key=s['history'];cache.setdefault(key,maps(key)) if key not in cache else None
    gates,L=cache[key];top,second=s['top'],s['second']
    expected=gates+[tuple(int(q==a)-int(q==b) for q in range(4)) for a,b in [(top,second)]+[(second,k) for k in range(4) if k not in(top,second)]]
    if s['kind']=='correct':
        for k in range(4):
            if k==second:continue
            row=[a-b for a,b in zip(L[second],L[k])]
            if any(row):expected.append(canonical(row))
            else:assert k>second
    elif s['kind']=='wrong':
        k=s['competitor'];row=[a-b for a,b in zip(L[k],L[second])]
        if any(row):expected.append(canonical(row))
        else:assert k<second
    rows=[tuple(map(int,r)) for r in s['rows']];assert rows==expected
    c=s['certificate'];counts[c['status']]+=1
    if c['status']=='feasible':
        x=list(map(F,c['witness']));assert all(sum(a*b for a,b in zip(r,x))>0 for r in rows)
        h=[F(0)]*4;history=''
        for u in x:
            z=[sum(a*b for a,b in zip(r,h))+v*u for r,v in zip(wh,wi)]
            history+=''.join('1' if v>0 else '0' for v in z);h=[max(F(0),v) for v in z]
        scores=[sum(a*b for a,b in zip(r,h)) for r in wo];pred=max(range(4),key=lambda j:scores[j]);order=sorted(range(4),key=lambda j:x[j])
        assert history==key and order[-1]==top and order[-2]==second
        if s['kind']=='correct':assert pred==second
        if s['kind']=='wrong':assert pred!=second
        forward_checks+=1
    elif c['status']=='infeasible':
        dual={int(i):F(v) for i,v in c['dual'].items()};assert min(dual.values())>=0 and sum(dual.values())==1
        assert all(sum(v*rows[i][q] for i,v in dual.items())==0 for q in range(4))
out={'systems':len(d['systems'])-skip,'unchanged_previously_verified_systems':skip,'statuses':dict(counts),'direct_fraction_forwards':forward_checks,'distinct_history_maps':len(cache),
     'panel_complete':d['complete'],'lp_cap_stop':d.get('stop'),'cpu_seconds':time.process_time()-start,
     'scope':'stored new row systems and exact witnesses verified independently; no conditioned masses computed'}
name='joint_certificates_phase2_review_results.json' if skip else 'joint_certificates_review_results.json'
Path(__file__).with_name(name).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

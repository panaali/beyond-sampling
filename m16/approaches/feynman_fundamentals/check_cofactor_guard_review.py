"""Independent exact elimination and source-category audit of saved controls."""
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/cofactor3_guard')
a=json.loads((P/'results.json').read_text())
assert a['complete'] and len(a['rows'])==25
for name,h in a['source_sha256'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
def determinant(A):
    B=[list(map(F,row)) for row in A]; out=F(1)
    for k in range(3):
        pivot=next((j for j in range(k,3) if B[j][k]),None)
        if pivot is None:return F(0)
        if pivot!=k:B[k],B[pivot]=B[pivot],B[k];out=-out
        v=B[k][k];out*=v
        for j in range(k+1,3):
            t=B[j][k]/v
            for z in range(k+1,3):B[j][z]-=t*B[k][z]
    return out
statuses=Counter(); lost=[]; fast_cross=0
for row in a['rows']:
    A=[[float.fromhex(x) for x in r] for r in row['input_hex']]
    flat=sum(A,[]); r=row['result']; c=Counter(to_float=9,isfinite=9)
    finite=all(map(math.isfinite,flat)); statuses[r['status']]+=1
    if not finite:
        assert r['status']=='unknown_nonfinite' and dict(c)==r['counts'];continue
    q=determinant(A); assert q==F(*map(int,row['exact'])) and row['enclosed']
    lo,hi,point=(float.fromhex(r[k]) for k in ('lower','upper','point'))
    assert (lo==-math.inf or F(lo)<=q) and (hi==math.inf or F(hi)>=q)
    c['abs']=9
    for x in flat:
        z=abs(x);c['comparison']+=1
        if z:
            c['comparison']+=1
            if 2.**-100<=z:c['comparison']+=1
    eligible=all(x==0 or 2.**-100<=abs(x)<=2.**100 for x in flat)
    if eligible:
        assert r['status']=='fast_enclosure'
        c.update(float_mul=19,float_sub=4,float_add=6,comparison=1)
        E=float.fromhex(r['error_allowance'])
        Phat=float.fromhex(r['permanent'])
        assert F(E)==F(Phat)*F(1,2**50)
        assert abs(F(point)-q)<=F(E)
        if E==0:c['comparison']+=1;assert q==point==lo==hi==0
        else:
            c.update(float_sub=1,float_add=1,nextafter=2)
            assert lo==math.nextafter(point-E,-math.inf)
            assert hi==math.nextafter(point+E,math.inf)
        fast_cross+=lo<=0<=hi
    else:
        assert r['status']=='exact_fallback' and q==F(*map(int,r['exact']))
        c.update(float_to_exact=9,exact_mul=9,exact_sub=4,exact_add=1,exact_to_float=1,isfinite=1)
        try: expected=float(q)
        except OverflowError:
            expected=math.inf if q>0 else -math.inf;c['exact_comparison']+=1
        assert point==expected
        if math.isfinite(point):c['nextafter']=2
        else:c['comparison']+=1
    assert dict(c)==r['counts'],row['name']
    if point==0 and q:lost.append(row['name'])
u=F(1,2**53); gap=8*(1-u)**5*(1-5*u)-5
assert gap>0 and gap==F(*map(int,a['gamma_inequality_gap']))
assert statuses==dict(fast_enclosure=17,exact_fallback=5,unknown_nonfinite=3)
out=dict(complete=True,controls=25,statuses=dict(statuses),all_exact_elimination_and_ledgers_pass=True,
         fast_intervals_containing_zero=fast_cross,zero_points_nonzero_determinants=lost,
         source_sha256=a['source_sha256'],new_primitive_mass_model_calls=0,
         scope='Stored binary64 determinant enclosure under the declared nonfused round-to-nearest tree. Poor point estimates and cross-zero intervals remain explicit; no relative point, mass or native-cost certificate.')
(HERE/'cofactor_guard_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

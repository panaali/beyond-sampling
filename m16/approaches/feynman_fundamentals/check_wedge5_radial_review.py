"""Independent exact saved Gram/KKT audit and high-precision tail contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,itertools,json,time
import mpmath as mp
mp.mp.dps=180
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/wedge5_controls/radial_bound')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def m(x):
    x=F(x);return mp.mpf(x.numerator)/x.denominator
def determinant(A):
    # Independent 120-term permutation expansion, not author elimination.
    total=F(0)
    for perm in itertools.permutations(range(5)):
        v=F((-1)**sum(perm[i]>perm[j] for i in range(5) for j in range(i+1,5)))
        for i,j in enumerate(perm):v*=A[i][j]
        total+=v
    return total
start=time.process_time();d=read(P/'results.json');s=read(P/'summary.json');protocol=read(P/'protocol.json')
assert d['complete'] and s['complete'] and d['protocol']==protocol
assert sha(P/'results.json')==s['results_sha256']
for n,h in d['source_sha256'].items():assert sha(P/n)==h
assert sha(Path(protocol['directed_source']))==protocol['directed_sha256']
assert sha(Path(protocol['input_results']))==protocol['input_sha256']
prior=read(Path(protocol['input_results']));fixtures={f['name']:f for f in prior['fixtures']}
certs={c['fixture']:c for c in d['certificates']};assert len(certs)==len(fixtures)==11
exact=Counter();directed=Counter();modes=Counter()
for name,c in certs.items():
    A=[[F(float.fromhex(x)) for x in row] for row in fixtures[name]['rays_hex']]
    G=[[sum((x*y for x,y in zip(u,v)),F(0)) for v in A] for u in A]
    assert G==[list(map(F,row)) for row in c['gram']]
    det=determinant(A);assert det==F(c['det'])==F(fixtures[name]['exact_stored_determinant']) and det!=0
    a,b=F(c['a']),F(c['b']);assert 0<a<=b==max(G[i][i] for i in range(5))
    modes[c['mode']]+=1
    if c['mode']=='acute':
        assert all(G[i][j]>=0 for i in range(5) for j in range(5))
        assert a==min(G[i][i] for i in range(5))/5
        expected=dict(multiply=160,add=125,divide=11,subtract=30)
    else:
        assert c['mode']=='exact_kkt' and c['examined_faces']==c['feasible_faces']==31
        y=list(map(F,c['y']));assert sum(y)==1 and min(y)>=0
        Gy=[sum(G[i][j]*y[j] for j in range(5)) for i in range(5)]
        slack=[x-a for x in Gy]
        assert slack==list(map(F,c['slack'])) and min(slack)>=0 and all(x*z==0 for x,z in zip(y,slack))
        assert sum(x*z for x,z in zip(y,Gy))==a
        # Full-row elimination for every nonempty support: sum n(n+1)=320,
        # sum n(n-1)(n+1)=720; feasibility and final KKT arithmetic included.
        expected=dict(multiply=910,add=235,divide=441,subtract=755)
    assert c['counts']==expected;exact.update(c['counts'])
assert modes==Counter(acute=6,exact_kkt=5)
assert len(d['rows'])==33 and len(d['controls'])==9
seen=set();qualified=[]
for row in d['controls']+d['rows']:
    control='control' in row
    c=protocol['additional_controls'][row['control']] if control else certs[row['fixture']]
    a,b=F(row['a_expanded']),F(row['b_expanded']);assert 0<a<=F(c['a'])<=F(c['b'])<=b
    n=row['order'];N=n-2;assert n in (6,10,16) and row['degree']==N
    mid=(m(a)+m(b))/2;r=(m(b)-m(a))/(m(b)+m(a))
    assert m(row['r_interval'][0])<=r<=m(row['r_interval'][1])
    # mpmath hypergeometric rising factorial avoids the author's recurrence.
    partial=sum(mp.rf(mp.mpf('2.5'),k)*r**k/mp.factorial(k) for k in range(N+1))
    tail=mp.mpf(0) if a==b else m(a)**(-mp.mpf('2.5'))-mid**(-mp.mpf('2.5'))*partial
    assert m(row['raw_tail_interval'][0])<=tail<=m(row['raw_tail_interval'][1])
    assert F(row['tail_interval'][0])==max(F(0),F(row['raw_tail_interval'][0]))
    assert F(row['tail_interval'][1])==F(row['raw_tail_interval'][1])>=0
    value=abs(m(c['det']))*tail/(32*mp.pi**2)
    assert m(row['lower_formula_endpoint'])<=value<=m(row['upper_bound'])
    assert row['qualifies']==(F(row['upper_bound'])<=F(1,10**10))
    expected=(dict(directed_divide=8,directed_multiply=2) if a==b else
              dict(directed_divide=16+2*N,directed_multiply=16+4*N,directed_add=4+2*N,
                   directed_subtract=4,correctly_rounded_sqrt=4,decimal_adjacent=4))
    assert row['counts']==expected and row['coefficient_counts']==({} if a==b else {'rational_multiply':N})
    if not control:
        key=(row['fixture'],n);assert key not in seen;seen.add(key)
        old=next(z for z in prior['rows'] if z['fixture']==row['fixture'] and z['arm']==f'duffy{n}')
        assert row['source_value_hex']==old['returned_hex'] and row['source_absolute_error']==old['absolute_error']
        assert row['observed_error_within_ideal_bound']==(F(old['absolute_error'])<=F(row['upper_bound']))
        directed.update(row['counts'])
        if row['qualifies']:qualified.append(key)
assert seen==set(itertools.product(fixtures,(6,10,16)))
assert set(qualified)==set(itertools.product(('narrow35','narrow48'),(6,10,16)))
assert dict(exact)==s['exact_counts'] and dict(directed)==s['directed_counts'] and s['qualified']==6
out=dict(complete=True,exact_certificates=11,certificate_modes=dict(modes),bounds=33,controls=9,
         qualified=qualified,unresolved=27,exact_counts=dict(exact),directed_counts=dict(directed),
         results_sha256=sha(P/'results.json'),review_cpu=time.process_time()-start,new_mass_calls=0,
         scope='Exact stored Gram/determinant/KKT certificates, source-directed interval formula and independent 180-digit tail checks. Ideal fixed positive quadrature truncation only; floating nodes/determinants/kernel and log quadrature remain excluded. Loose bounds are not observed integration failures.')
(HERE/'wedge5_radial_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

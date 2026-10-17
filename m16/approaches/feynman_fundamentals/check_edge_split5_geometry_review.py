"""Independent saved exact positive-edge split certificates; no numerical kernels."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,itertools,json
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_910/review/edge_split5')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def matrix(h):return [[F(float.fromhex(x)) for x in r] for r in h]
def det(A):
    total=F(0)
    for perm in itertools.permutations(range(5)):
        t=F((-1)**sum(perm[i]>perm[j] for i in range(5) for j in range(i+1,5)))
        for i,j in enumerate(perm):t*=A[i][j]
        total+=t
    return total
d=read(P/'geometry.json');p=read(P/'protocol.json')
assert d['complete'] and d['protocol_sha256']==sha(P/'protocol.json') and d['source_sha256']==sha(P/'geometry.py')
for path,h in p['pins'].items():assert sha(Path(path))==h
prior=read(Path(p['prior'])/'results.json');fixtures={f['name']:f for f in prior['fixtures']}
assert len(d['controls'])==7 and [r['name'] for r in d['controls']]==p['guard_controls']
assert all(r['status']=='completed' and r['pass_control'] and r['expected']==r['result'] for r in d['controls'])
assert [r['name'] for r in d['rows']]==p['fixture_names'] and d['accepted']==11
for r in d['rows']:
    assert r['status']=='accepted' and r['parent_hex']==fixtures[r['name']]['rays_hex']
    A=matrix(r['parent_hex']);V=list(map(lambda x:F(float.fromhex(x)),r['candidate_hex']))
    coeff=list(map(F,r['coefficients']));a,b=coeff[:2];assert a>0 and b>0 and coeff[2:]==[F(0)]*3
    assert [a*A[0][j]+b*A[1][j] for j in range(5)]==V
    D=det(A);assert D==F(r['parent_determinant'])!=0
    for i,hh in enumerate(r['children_hex']):
        C=matrix(hh);expected=[list(row) for row in A];expected[i]=V
        assert C==expected and det(C)==coeff[i]*D==F(r['child_determinants'][i])
    w=list(map(F,r['section_dual']));assert [sum(x*y for x,y in zip(w,row)) for row in A]==[F(1)]*5
    vdot=sum(x*y for x,y in zip(w,V));assert vdot==a+b
    assert list(map(F,r['section_ratios']))==[a/(a+b),b/(a+b)] and sum(map(F,r['section_ratios']))==1
    raw=list(map(lambda x:F(float.fromhex(x)),r['raw_sum_hex']));norm=F(float.fromhex(r['norm_hex']));assert norm>0
    exactsum=[x+y for x,y in zip(A[0],A[1])]
    assert exactsum==list(map(F,r['exact_sum'])) and [x-y for x,y in zip(raw,exactsum)]==list(map(F,r['addition_error']))
    assert [v-x/norm for v,x in zip(V,raw)]==list(map(F,r['normalization_error']))
    assert [float(x/norm).hex() for x in raw]==r['candidate_hex']
    assert r['floating_setup_named']==dict(addition=5,norm_multiply=5,norm_add=4,sqrt=1,divide=5)
out=dict(complete=True,accepted_exact_partitions=11,rejection_controls=7,child_determinants_checked=22,
         geometry_sha256=sha(P/'geometry.json'),new_mass_kernel_calls=0,
         scope='Independent permutation determinants, actual binary positive-edge coefficients, common dual and section ratios, stored rounding differences. Exact cone additivity follows; section ratios are not Gaussian mass fractions. Rejected-path rational costs and compiled setup internals remain excluded.')
(HERE/'edge_split5_geometry_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

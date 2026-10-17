"""Independent angular references and saved fixed-rule evidence only."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,ast,math
import mpmath as mp
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/wedge5_controls')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(P/'results.json');s=read(P/'summary.json');p=read(P/'protocol.json')
assert a['complete'] and a['protocol']==p and a['pins_sha256']==sha(P/'pins.json')
for path,h in read(P/'pins.json').items():assert sha(Path(path))==h
assert sha(P/'estT_snapshot.py')==p['source_sha256']
t=ast.parse((P/'estT_snapshot.py').read_text());body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('set_rules5','_orth5')]
assert hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest()==a['extracted_ast_sha256']
mp.mp.dps=160
def M(q):return mp.mpf(q.numerator)/q.denominator
refs={};dets={};reference_diff=[]
assert len(a['fixtures'])==11
for r in a['fixtures']:
    R=[[F(float.fromhex(x)) for x in row] for row in r['rays_hex']]
    assert len(R)==5 and all(len(x)==5 for x in R)
    for i in (0,2,3,4):assert R[i]==[F(i==j) for j in range(5)]
    c,ss=R[1][:2];assert ss>0 and R[1][2:]==[0,0,0]
    assert F(r['exact_stored_determinant'])==ss and F(r['exact_pair_dot'])==c
    assert F(r['stored_second_ray_norm2'])==c*c+ss*ss
    # Independent bivariate angle via normalized dot, rather than atan2.
    angle=mp.acos(M(c)/mp.sqrt(M(c*c+ss*ss)));ref=angle/(16*mp.pi)
    assert abs(ref-mp.mpf(r['reference120']))<mp.mpf('1e-115')
    reference_diff.append(float(abs(ref-mp.mpf(r['reference120']))))
    refs[r['name']]=ref;dets[r['name']]=ss
for k in p['ks']:assert dets[f'narrow{k}']==dets[f'wide{k}']
assert len(a['rows'])==66 and len({(r['arm'],r['fixture']) for r in a['rows']})==66
rows={(r['arm'],r['fixture']):r for r in a['rows']};arms={r['name']:r for r in p['arms']}
summary={}
for arm,z in arms.items():
    rr=[r for r in a['rows'] if r['arm']==arm];assert len(rr)==11
    for r in rr:
        assert r['finite'] and not r.get('exception') and r['source_simplices']==1
        value=float.fromhex(r['returned_hex']);assert value==r['value']
        error=mp.mpf(value)-refs[r['fixture']]
        assert math.isclose(float(abs(error)),r['absolute_error'],rel_tol=2e-15,abs_tol=1e-120)
        assert math.isclose(float(abs(error/refs[r['fixture']])),r['relative_error'],rel_tol=2e-15)
        assert r['absolute_pass']==(abs(error)<=mp.mpf('1e-10'))
        assert r['relative_pass']==(abs(error/refs[r['fixture']])<=mp.mpf('1e-6'))
        thin=z['threshold']=='infinity' or (z['threshold']!=0 and dets[r['fixture']]<F(z['threshold']))
        ops=125+40*z['thin']**4 if thin else 1540+1340*z['log']
        assert ops==r['source_proxy_ops']
    q=dict(absolute_pass=sum(r['absolute_pass'] for r in rr),relative_pass=sum(r['relative_pass'] for r in rr),
           max_absolute_error=max(r['absolute_error'] for r in rr),max_relative_error=max(r['relative_error'] for r in rr),
           source_proxy_ops=sum(r['source_proxy_ops'] for r in rr))
    for key,value in q.items():assert value==s['arms'][arm][key]
    summary[arm]=q
for name in refs:
    chosen='duffy6' if dets[name]<F(1,1_000_000) else 'log128'
    assert rows['source',name]['returned_hex']==rows[chosen,name]['returned_hex']
out=dict(complete=True,fixtures=11,saved_calls=66,independent_acos_reference_max_difference=max(reference_diff),
         arms=summary,results_sha256=sha(P/'results.json'),new_mass_model_or_random_calls=0,
         theorem='Any fixed finite strictly interior simplex rule gives O(sin epsilon) on these wide wedges, while true mass tends to 1/16; determinant-only thin routing has no uniform correctness guarantee.',
         scope='Analytic product controls and ideal-rule obstruction only. Forced log passes these absolute tolerances but has no general error certificate. No trained-corpus conclusion or adaptive rule choice.')
(HERE/'wedge5_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

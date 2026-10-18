"""Independent saved-only T200 audit. No author imports or cap calls."""
from fractions import Fraction as F
from pathlib import Path
import hashlib, json, math, subprocess, time, traceback

HERE = Path(__file__).resolve().parent
AUTHOR = Path('/workspace/work/recover-small_h4_4/review/conservative_cap_bound')
REPO = HERE.parents[1]
NAMES = ['orthant2','orthant3','narrow3','zero_axis2','orthant4','orthant5',
         'zero_ray3','unsupported6','nonfinite3','bit_limit3']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git',*a],cwd=REPO)
def dot(x,y): return sum((a*b for a,b in zip(x,y,strict=True)), F(0))
def bits(x):
    x=F(x)
    return max(abs(x.numerator).bit_length(), x.denominator.bit_length())

def geometry(R):
    """Independent direct formula, plus exact source-intermediate bit inventory."""
    n,d=len(R),len(R[0]); seen=[v for row in R for v in row]
    axis=[sum((r[j] for r in R),F(0)) for j in range(d)]
    for j in range(d): seen += [sum((R[i][j] for i in range(k)),F(0)) for k in range(1,n+1)]
    def observed_dot(x,y):
        products=[a*b for a,b in zip(x,y,strict=True)]
        seen.extend(products)
        seen.extend(sum(products[:k],F(0)) for k in range(1,len(products)+1))
        return sum(products,F(0))
    norm=observed_dot(axis,axis)
    ratios=[]
    for row in R:
        rn=observed_dot(row,row); ad=observed_dot(axis,row)
        assert norm>0 and rn>0 and ad>0
        ratios.append(ad*ad/(norm*rn)); seen.extend([ad*ad,norm*rn,ratios[-1]])
    q=min(ratios); delta=1-q; seen.append(delta)
    assert 0<q<=1
    u=None
    if d in (2,4):
        # Floor first, then compare the exact scaled square. Different from
        # the author's ceil-divide then integer-ceil-square-root algorithm.
        scaled=delta.numerator*2**80
        m=math.isqrt(scaled//delta.denominator)
        m += int(m*m*delta.denominator<scaled)
        u=F(m,2**40)
        assert u*u>=delta and (m==0 or (u-F(1,2**40))**2<delta)
        seen.extend([scaled,u])
    if d==2: raw=u/2
    elif d==3: raw=delta/2
    elif d==4: seen.append(delta*u); raw=F(2,3)*delta*u
    else: seen.append(delta*delta); raw=F(3,4)*delta*delta
    seen.append(raw)
    return axis,q,delta,min(F(1,2),raw),max(map(bits,seen))

def main():
    start=time.process_time()
    gate=json.loads((AUTHOR/'gate.json').read_text())
    sourcegate=json.loads((HERE/'source_gate.json').read_text())
    for p,h in {**sourcegate['pins'],**gate['pins']}.items(): assert sha(Path(p))==h,p
    for filename,commit in [('gate.json','1ed5296b'),('results.json','88ff3af7'),
                            ('progress.jsonl','88ff3af7')]:
        assert git('show',f'{commit}:review/conservative_cap_bound/{filename}')==(AUTHOR/filename).read_bytes()
    for commit in ['7b965425','1ed5296b','88ff3af7','15574e6f','d84f5236']:
        subprocess.run(['git','merge-base','--is-ancestor',commit,'7204c26d'],cwd=REPO,check=True)
    subprocess.run(['git','merge-base','--is-ancestor','1ed5296b','88ff3af7'],cwd=REPO,check=True)
    result=json.loads((AUTHOR/'results.json').read_text())
    progress=[json.loads(s) for s in (AUTHOR/'progress.jsonl').read_text().splitlines()]
    old=json.loads(Path(gate['old_cap_result']).read_text())['controls']
    assert result['complete'] and result['gate_sha256']==sha(AUTHOR/'gate.json')
    assert [r['name'] for r in result['rows']]==NAMES and len(progress)==22
    assert progress[0]=={'stage':'started','gate_sha256':sha(AUTHOR/'gate.json')}
    assert progress[-1]=={'stage':'completed','result_sha256':sha(AUTHOR/'results.json')}
    assert result['author_cap_calls']==result['model_geometry_mass_reference_MC_calls']==0
    checked=[]
    for i,row in enumerate(result['rows']):
        name=row['name']; saved=row['result']; rays=row['rays']; n,d=len(rays),len(rays[0])
        assert progress[2*i+1]=={'stage':'fixture_started','name':name}
        assert progress[2*i+2]=={'stage':'fixture_completed','name':name,'result':saved}
        if i<4: assert row['rays']==old[i]['rays'] and name==old[i]['name']
        expected_peak=None
        if saved['fallback'] is None:
            R=[[F(v) for v in r] for r in rays]
            axis,q,delta,bound,expected_peak=geometry(R)
            assert saved['details']==dict(dimension=d,rays=n,q=str(q),delta=str(delta),axis=list(map(str,axis)))
            counts=dict(binary64_to_fraction=n*d,rational_add=d*(3*n+1),
                        rational_mul=d+2*n*d+2*n+(2 if d in (4,5) else 0),
                        rational_compare=2+5*n,rational_div=n+int(d in (2,3)),rational_sub=1)
            if d in (2,4):
                counts.update({k:1 for k in ['integer_shift','integer_divmod','integer_compare','integer_isqrt',
                                            'integer_square_compare','rational_construct','rational_square_compare']})
            if name.startswith('orthant'):
                assert rays==[[float(a==b) for b in range(d)] for a in range(d)]
                mass=F(1,2**d)
                assert bound>=mass and F(row['exact_cone_mass'])==mass and F(row['bound_over_mass'])==bound/mass
            else:
                e=F(1,2**30)
                assert R==[[1,0,0],[1,e,0],[1,0,e]]
                assert 0<F(row['strict_mass_lower'])==e*e/41472<bound
                assert row['old_cap']==old[2]['value']==0
        else:
            bound=F(1)
            fallback={
                'zero_axis2':('zero_axis',dict(binary64_to_fraction=8,rational_add=10,rational_mul=2,rational_compare=1),1),
                'zero_ray3':('zero_ray',dict(binary64_to_fraction=6,rational_add=15,rational_mul=9,rational_compare=3),1),
                'unsupported6':('unsupported_dimension',{},0),
                'nonfinite3':('nonfinite_or_type',{},0),
                'bit_limit3':('bit_limit',dict(binary64_to_fraction=9,rational_add=11,rational_mul=2),2)}
            reason,counts,expected_peak=fallback[name]
            assert saved['fallback']==reason and saved['details']==dict(dimension=d,rays=n)
            if name=='zero_axis2': assert all(sum(r[j] for r in rays)==0 for j in range(d))
            if name=='zero_ray3': assert rays==[[0.,0.,0.],[1.,0.,0.]]
            if name in ('unsupported6','bit_limit3'): assert rays==[[float(a==b) for b in range(d)] for a in range(d)]
            if name=='nonfinite3': assert rays==[['nan',0.,0.]]
        assert row['options']==({'maxbits':1} if name=='bit_limit3' else {})
        ledger=saved['ledger']
        assert ledger==dict(counts=counts,counted_steps=sum(counts.values()),max_fraction_bits=expected_peak,
                           bit_limit=1 if name=='bit_limit3' else 8192,step_limit=128000),name
        assert F(saved['bound'])==bound
        outward=saved['outward']; assert outward.hex()==saved['outward_hex'] and 0<=bound<=F(outward)<=1
        nearest=float(bound); changed=int(F(nearest)<bound)
        expected=math.nextafter(nearest,math.inf) if changed else nearest
        assert outward.hex()==expected.hex() and saved['conversion']==dict(fallback=None,nextafter=changed)
        checked.append(dict(name=name,bound=str(bound),peak_bits=expected_peak,macro_steps=sum(counts.values())))
    assert math.isclose(sum(r['cpu_seconds'] for r in result['rows']),result['cpu_seconds'],rel_tol=0,abs_tol=1e-15)
    return dict(status='PASS_SAVED_ONLY',rows=checked,controls=10,progress_records=22,
                author_cpu_seconds=result['cpu_seconds'],review_cpu_seconds=time.process_time()-start,
                pins={str(p):sha(p) for p in [AUTHOR/'results.json',AUTHOR/'progress.jsonl',AUTHOR/'gate.json',HERE/'source_gate.json',Path(__file__)]},
                publication_order='Deviation retained: immutable gate committed before run, remote publication after run. Source/theory already shared. Ancestry and bytes verified; publication time relies on preserved author disclosure.',
                author_imports=0,author_cap_calls=0,model_mass_geometry_reference_MC_calls=0,
                scope='Exact represented-hull primitive and saved controls, not estimator correctness, trained usefulness or native cost.')

if __name__=='__main__':
    target=HERE/'saved_review.json'
    assert not target.exists()
    try: out=main()
    except BaseException:
        with (HERE/'saved_review_failure.txt').open('x') as f: f.write(traceback.format_exc())
        raise
    with target.open('x') as f: json.dump(out,f,indent=2); f.write('\n')
    print(json.dumps({'status':out['status'],'controls':out['controls'],'cpu':out['review_cpu_seconds']}))

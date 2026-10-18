"""Fixed exact synthetic controls; no trained domains or numerical estimator."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,resource,signal,sys,time
from primitive import certify,pack
from frozen_canonical import direct,from_key
from inherited_fixtures import fixtures as inherited,net
HERE=Path(__file__).resolve().parent

def fixtures():
    rows=inherited()
    for r in rows:
        r['inherited_expected']=r['expected']
        if r['id']=='lower_dimensional_linear_unknown':r['expected']=True
    z2=net([], [[]],[[0,0]]);z3=net([], [[]],[[0,0,0]])
    line=[[1,2,3],[-1,-2,-3]];plane=[[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]];half=[[1,0],[-1,0],[0,1]]
    def add(name,a,b,ra,rb,want):rows.append(dict(id=name,a=a,b=b,rays_a=ra,rays_b=rb,expected=want,inherited_expected=None))
    add('zero_span',net([[1,2]],[[3]],[[4,5]]),z2,[[0,0]],[[0,0],[0,0]],True)
    add('coincident_after_projection',net([[1,0,1]],[[1]]),net([[1,0,0]],[[1]]),plane,list(reversed(plane)),True)
    add('noncoordinate_rank_one',net([[2,0,0]],[[1]]),net([[0,1,0]],[[1]]),line,[[0,0,0]]+list(reversed(line)),True)
    add('rank_two_pivot_swaps',net([], [[]],[[1,0,0]]),z3,[[0,0,0],[0,1,0],[0,0,1],[0,1,1]],[[0,0,1],[0,1,0]],True)
    add('lineality_mixed_unequal',net([[1,0]],[[1]]),z2,half,half,False)
    add('vector_supporting_plane',net([[0,1]],[[1],[-2]]),net([], [[],[]],[[0,1],[0,-2]]),half,[[0,1],[1,1]],True)
    f=net([[1,0],[0,1],[1,1]],[[1,1,-1]])
    add('nonconvex_union_equal',f,z2,[[2,1]],[[-1,-2]],True)
    add('convexified_union_unequal',f,z2,[[2,1],[-1,-2]],[[2,1],[-1,-2]],False)
    assert len(rows)==24 and sum(r['expected'] for r in rows)==19
    return rows

def sample_points(R,coords):
    m=len(R);n=len(R[0]);d=len(coords[0]);weights=[[0]*m]+[[int(i==j) for i in range(m)] for j in range(m)]+[[1]*m,list(range(1,m+1))]
    points={}
    for w in weights:
        x=tuple(sum((F(v)*r[j] for v,r in zip(w,R,strict=True)),F(0)) for j in range(n))
        z=tuple(sum((F(v)*r[j] for v,r in zip(w,coords,strict=True)),F(0)) for j in range(d))
        if x in points:assert points[x]==z
        points[x]=z
    return sorted(points.items())

def guardcases():
    a=net([[1,0]],[[1]]);z=net([], [[]],[[0,0]]);r=[[1,0]]
    return [('empty_hull',a,z,[],r),('ragged_hull',a,z,[[1,0],[1]],r),('zero_width',a,z,[[]],r),('form_width',net([[1,0,0]],[[1]]),z,r,r),('coefficient_width',net([[1,0]],[[1,2]]),z,r,r),('output_dimension',a,net([], [[],[]]),r,r),('nonfinite_form',net([[float('inf'),0]],[[1]]),z,r,r),('nonfinite_ray',a,z,[[float('nan'),0]],r),('hull_dimension_mismatch',net([], [[]]),net([], [[]]),r,[[1,0,0]])]

def main():
    out=HERE/'controls.json';progress=HERE/'progress.jsonl';assert not out.exists() and not progress.exists()
    resource.setrlimit(resource.RLIMIT_CPU,(19,20))
    def timeout(*args):raise RuntimeError('soft CPU cap19')
    signal.signal(signal.SIGXCPU,timeout)
    start=time.process_time();result=dict(status='started',rows=[],guards=[])
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(pack(row),allow_nan=False)+'\n')
    def save():out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n')
    record(dict(stage='attempt'));save()
    try:
        p=json.loads((HERE/'protocol.json').read_text());assert isinstance(p['source_gate'],dict)
        gate=p['source_gate'];assert hashlib.sha256(Path(gate['artifact']).read_bytes()).hexdigest()==gate['sha256']
        assert str(Path(sys.executable).resolve())==p['runtime_python']
        for path,want in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==want,path
        result['protocol_sha256']=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest();calls=0
        for c in fixtures():
            record(dict(stage='case_attempt',id=c['id']));answer=certify(c['a'],c['b'],c['rays_a'],c['rays_b']);checks=[]
            for dom in answer['domains']:
                a,b=dom['original_a'],dom['original_b']
                for x,z in sample_points(dom['rays'],dom['span']['coordinates']):
                    va=direct(a['B'],a['V'],x,a['L']);vb=direct(b['B'],b['V'],x,b['L']);calls+=2;diff=[u-v for u,v in zip(va,vb,strict=True)]
                    if dom['span']['dimension']:
                        pa,pb=dom['projected_a'],dom['projected_b'];qa=direct(pa['B'],pa['V'],z,pa['L']);qb=direct(pb['B'],pb['V'],z,pb['L']);calls+=2
                        canonical=from_key(dom['canonical']['key'],z)
                    else:qa=[F(0)]*len(va);qb=[F(0)]*len(vb);canonical=[F(0)]*len(diff)
                    passed=va==qa and vb==qb and diff==canonical and (not dom['certified'] or not any(diff))
                    checks.append(dict(x=x,z=z,a=va,b=vb,projected_a=qa,projected_b=qb,difference=diff,canonical=canonical,passed=passed))
            passed=answer['certified']==c['expected'] and all(r['passed'] for r in checks)
            row=dict(fixture=c,result=answer,point_checks=checks,passed=passed);result['rows'].append(row);record(dict(stage='case_result',row=row));save();assert passed,c['id']
        for name,a,b,ra,rb in guardcases():
            record(dict(stage='guard_attempt',id=name));rejected=False;error=None
            try:certify(a,b,ra,rb)
            except (ValueError,OverflowError) as exc:rejected=True;error=type(exc).__name__
            row=dict(id=name,passed=rejected,error=error);result['guards'].append(row);record(dict(stage='guard_result',row=row));save();assert rejected,name
        assert len(result['rows'])==24 and len(result['guards'])==9 and sum(r['result']['certified'] for r in result['rows'])==19
        result.update(status='complete',synthetic_direct_calls=calls,new_trained_model_geometry_mass_MC_calls=0,all_pass=True);record(dict(stage='complete'))
    except BaseException as exc:
        result.update(status='failed',exception=repr(exc));record(dict(stage='failure',exception=repr(exc)));raise
    finally:result['cpu_seconds']=time.process_time()-start;save()
if __name__=='__main__':main()

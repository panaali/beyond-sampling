"""Sixteen frozen exact domain pairs and four rejection guards."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
from primitive import certify,pack
from canonical import direct,from_key
H=Path(__file__).resolve().parent

def net(B,V,L=None):
    n=dict(B=B,V=V)
    if L is not None:n['L']=L
    return n

def fixtures():
    e=[1,0];f=[0,1];pos=[e,f];neg=[[-1,0],[0,-1]];full=[e,f,[-1,0],[0,-1]];zero=net([], [[]],[[0,0]]);relu=net([e],[[1]]);linear=net([], [[]],[e]);rows=[]
    def add(name,a,b,ra,rb,expected):rows.append(dict(id=name,a=a,b=b,rays_a=ra,rays_b=rb,expected=expected))
    add('identity_crossing',relu,relu,full,full,True)
    add('positive_linear',relu,linear,pos,[e,[1,1]],True)
    add('negative_zero',relu,zero,neg,[[-1,0],[-1,1]],True)
    add('negative_orientation_active',net([[-1,0]],[[1]]),net([], [[]],[[-1,0]]),neg,[[-1,0],[-1,1]],True)
    add('absolute_positive',net([e,[-1,0]],[[1,1]]),linear,pos,[e,[1,1]],True)
    add('absolute_negative',net([e,[-1,0]],[[1,1]]),net([], [[]],[[-1,0]]),neg,[[-1,0],[-1,1]],True)
    add('second_domain_blocks',relu,linear,pos,neg,False)
    add('zero_boundary_hinge',relu,zero,[f],[[0,-1]],True)
    add('lower_dimensional_linear_unknown',linear,zero,[f],[[0,-1]],False)
    add('negative_orientation_identity',net([[-1,0]],[[1]]),net([e],[[1]],[[-1,0]]),full,full,True)
    add('positive_hinge_rescale',net([[2,0]],[[F(1,2)]]),relu,full,full,True)
    add('first_index_full_ties',net([e],[[0],[0]]),net([f],[[0],[0]]),full,pos,True)
    add('shared_linear',net([], [[],[]],[[1,2],[-1,3]]),net([], [[],[]],[[1,2],[-1,3]]),full,neg,True)
    add('different_linear',linear,net([], [[]],[[2,0]]),pos,pos,False)
    add('vector_positive_linear',net([e],[[1],[2]]),net([], [[],[]],[[1,0],[2,0]]),pos,[e,[1,1]],True)
    add('different_hyperplane',relu,net([[1,1]],[[1]]),pos,pos,False)
    assert len(rows)==16;return rows

def points(rays):
    R=[list(map(F,r)) for r in rays];d=len(R[0]);weights=[[0]*len(R)]+[[int(i==j) for i in range(len(R))] for j in range(len(R))]+[[1]*len(R),list(range(1,len(R)+1))]
    return sorted({tuple(sum((F(w)*r[j] for w,r in zip(ws,R,strict=True)),F(0)) for j in range(d)) for ws in weights})

def main():
    out=H/'controls.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'protocol.json').read_text());assert p['source_gate']!='pending'
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    result=dict(complete=False,protocol_sha256=hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest());record(dict(stage='phase_attempt',protocol_sha256=result['protocol_sha256']));start=time.process_time()
    try:
        for f,sha in p['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha,f
        rows=[];calls=0
        for c in fixtures():
            record(dict(stage='case_attempt',id=c['id']));answer=certify(c['a'],c['b'],c['rays_a'],c['rays_b']);checks=[]
            for dom,rays in zip(answer['domains'],[c['rays_a'],c['rays_b']],strict=True):
                for x in points(rays):
                    a,b=c['a'],c['b'];va=direct(a['B'],a['V'],x,a.get('L'));vb=direct(b['B'],b['V'],x,b.get('L'));calls+=2
                    difference=[u-v for u,v in zip(va,vb,strict=True)];restricted=from_key(dom['canonical']['key'],x)
                    checks.append(dict(x=x,a=va,b=vb,difference=difference,restricted=restricted,passed=difference==restricted and (not dom['certified'] or not any(difference))))
            passed=answer['certified']==c['expected'] and all(x['passed'] for x in checks)
            row=dict(fixture=c,result=answer,point_checks=checks,passed=passed);rows.append(row);record(dict(stage='case_result',row=row));assert passed,'durable control failure'
        guards=[]
        base=net([[1,0]],[[1]]);other=net([], [[]],[[0,0]])
        guardcases=[('empty_hull',base,other,[],[[1,0]]),('ragged_hull',base,other,[[1,0],[1]],[[1,0]]),('coefficient_width',net([[1,0]],[[1,2]]),other,[[1,0]],[[1,0]]),('nonfinite',net([[float('inf'),0]],[[1]]),other,[[1,0]],[[1,0]])]
        for name,a,b,ra,rb in guardcases:
            record(dict(stage='guard_attempt',id=name));rejected=False;error=None
            try:certify(a,b,ra,rb)
            except (ValueError,OverflowError) as e:rejected=True;error=type(e).__name__
            row=dict(id=name,passed=rejected,error=error);guards.append(row);record(dict(stage='guard_result',row=row));assert rejected,'guard accepted malformed input'
        result.update(complete=True,rows=rows,guards=guards,cases=len(rows)+len(guards),synthetic_direct_evaluations=calls,new_trained_geometry_mass_MC_calls=0,cpu_seconds=time.process_time()-start,all_pass=all(r['passed'] for r in rows+guards))
        assert len(rows)==16 and len(guards)==4;record(dict(stage='complete'));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cases=20,cpu=result['cpu_seconds'],calls=calls)))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()

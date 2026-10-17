"""Independent DD geometry and arbitrary-precision scalar primitive checks."""
import argparse,hashlib,json,time,types,sys
from pathlib import Path
import numpy as np
import mpmath as mp


def module(name,path):
    raw=path.read_bytes();obj=types.ModuleType(name);exec(compile(raw,str(path),'exec'),obj.__dict__)
    return obj,hashlib.sha256(raw).hexdigest()


def reference_segment(a,b):
    """Independent antiderivative of (A*u*u+B*u+C)^(-5/2)."""
    with mp.workdps(100):
        x=[mp.mpf(float(v)) for v in a];y=[mp.mpf(float(v)) for v in b]
        delta=[v-u for u,v in zip(x,y)]
        aa=sum(v*v for v in delta);bb=2*sum(u*v for u,v in zip(x,delta));cc=sum(u*u for u in x)
        if aa==0:return float(cc**(-mp.mpf('2.5')))
        d=cc-bb*bb/(4*aa)
        if abs(d)<mp.mpf('1e-80'):
            r=mp.sqrt(cc);s=mp.sqrt(sum(v*v for v in y))
            if r==s:return float(r**-5)
            return float((r**-4-s**-4)/(4*(s-r)))
        def primitive(t):
            return t*(3*d+2*aa*t*t)/(3*d*d*(aa*t*t+d)**mp.mpf('1.5'))
        shift=bb/(2*aa)
        return float(primitive(shift+1)-primitive(shift))


def safe_outcomes(dd,cell,model,ops,base=70):
    """Independent first-index adapter; identical affine margins are not clipped."""
    R,Z,A,_=cell;T=len(model[2]);L=model[2]@A;good=[];skipped=0
    for j in range(T):
        q=(R,Z)
        for k in range(T):
            if k==j:continue
            normal=L[j]-L[k]
            if not np.any(normal):
                if k<j:q=None;skipped+=1;break
                continue
            q,_=dd.clip(q[0],q[1],normal,base+k,ops,want=(True,False))
            if q is None:break
        if q is None:continue
        states=[(q,0)]
        for k in range(T):
            if k==j:continue
            n=np.zeros(T);n[j]=1.;n[k]=-1.;next_states=[]
            for p,above in states:
                inside,outside=dd.clip(*p,n,base+T+k,ops)
                if inside is not None:next_states.append((inside,above))
                if outside is not None and above==0:next_states.append((outside,1))
            states=next_states
        good.extend(p for p,count in states if count==1)
    return good,skipped


def simplices(dd,R,Z,ops,T):
    pieces=[(R,Z)]
    for j in range(T):
        children=[]
        for r,z in pieces:
            sides=dd.clip(r,z,np.eye(T)[j],100+j,ops)
            children.extend(p for p in sides if p is not None and len(p[0])>=T)
        pieces=children
    out=[];zeros=0
    for r,z in pieces:
        r=r/np.sum(np.abs(r),axis=1)[:,None]
        for index in dd.triangulate(r,z,ops):
            v=r[list(index)].T
            if np.linalg.det(v)==0:zeros+=1;continue
            out.append(v)
    return out,zeros


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',default='/workspace/clean-neurons');parser.add_argument('--tolerance',type=float,default=1e-5);parser.add_argument('--dd-tol',type=float,default=1e-11);parser.add_argument('--output',default='results.json');a=parser.parse_args()
    app=Path(a.repo)/'m16/approaches';out=Path(__file__).with_name(a.output);start=time.process_time();deadline=start+112
    sys.path.insert(0,str(app/'small_h2_56'))
    dd,ddsha=module('independent_dd',app/'exact_ladder/cones.py');dd.TOL=a.dd_tol
    vol,vsha=module('volume',app/'small_h2_56/volume.py');sys.modules['volume']=vol
    seg,ssha=module('review_segment',app/'small_h2_56/segment.py')
    data=dict(source_sha256=dict(dd=ddsha,volume=vsha,segment=ssha),dd_tolerance=a.dd_tol,segment_controls=[],cases=[])
    def save():
        data['total_cpu']=time.process_time()-start;out.write_text(json.dumps(data,indent=2)+'\n')
    axis=np.eye(5)
    pairs=[(axis[0],axis[0]),(axis[0],axis[1]),(axis[0],2*axis[0]),(axis[0],axis[0]+1e-10*axis[1])]
    for k in range(12):
        x=1+np.sin(np.arange(5)+k)**2;y=1+np.cos(np.arange(5)*(k+1)+.3)**2
        pairs.append((x/x.sum(),y/y.sum()))
    pairs.extend([(pairs[7][0]*scale,pairs[7][1]*scale) for scale in (1e-8,1e8)])
    for x,y in pairs:
        expected=reference_segment(x,y);actual=float(seg.inverse_fifth_segment(x,y))
        data['segment_controls'].append(dict(a=x.tolist(),b=y.tolist(),actual=actual,reference=expected,relative_error=abs(actual-expected)/abs(expected)))
    save()
    import torch
    cp=Path(a.repo)/'work/checkpoints/H2_T5_seed0.pth';weights=torch.load(cp,weights_only=True,map_location='cpu')
    model=tuple(weights[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=(model[0].ravel(),*model[1:])
    data['checkpoint_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
    ops=dd.Ops();counts=[0]*5;correct=[];gates=[];zero_dets=0;tie_skips=0;max_affine=0.;cells=0;begin=time.process_time()
    for cell in dd.iter_cells(model,ops,counts):
        cells+=1;r,z,A,_=cell;x=r.sum(axis=0);x=x/np.linalg.norm(x)
        h=np.zeros(len(model[0]))
        for v in x:h=np.maximum(model[1]@h+model[0]*v,0)
        max_affine=max(max_affine,float(np.max(abs(h-A@x))))
        pieces,nzero=simplices(dd,r,z,ops,5);gates.extend(pieces);zero_dets+=nzero
        good,skips=safe_outcomes(dd,cell,model,ops);tie_skips+=skips
        for rr,zz in good:
            pieces,nzero=simplices(dd,rr,zz,ops,5);correct.extend(pieces);zero_dets+=nzero
        if time.process_time()>deadline:raise TimeoutError('review geometry CPU cap')
    correct=np.array(correct);gates=np.array(gates)
    row=dict(name='trained_seed0',gate_cells=cells,gate_counts_by_step=counts,correct_simplices=len(correct),gate_simplices=len(gates),zero_determinants=zero_dets,tie_candidates_rejected=tie_skips,max_affine_error=max_affine,dd_operations=ops.n,dd_operation_parts=ops.parts,geometry_cpu=time.process_time()-begin,gate_determinant_sum=float(np.abs(np.linalg.det(gates)).sum()),correct_determinant_sum=float(np.abs(np.linalg.det(correct)).sum()),ray_l1_error=float(np.max(abs(np.sum(abs(correct),axis=1)-1))),same_orthant_rays=bool(np.all(np.min(correct,axis=2)*np.max(correct,axis=2)>=-1e-14)))
    data['cases'].append(row);save()
    for name,rays in [('accuracy',correct),('partition',gates)]:
        begin=time.process_time();orders,bounds=vol.plan(rays,a.tolerance);planned=time.process_time()-begin
        result=seg.grouped_mass(rays,orders,bounds,deadline=deadline);result['planning_cpu']=planned;result['planning_and_integration_cpu']=time.process_time()-begin
        row[name]=result;save();print(name,result,flush=True)
    row['difference_from_T48']=row['accuracy']['estimate']-.2792373713559676
    row['partition_error']=row['partition']['estimate']-1.;save()
    print('geometry',row['gate_cells'],row['gate_determinant_sum'],row['max_affine_error'],flush=True)

if __name__=='__main__':main()

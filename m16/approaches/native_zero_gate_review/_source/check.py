"""Independent Python execution of T52's H2T5 outcome/orthant semantics."""
import argparse,hashlib,importlib.util,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from geometry_native import geometry,rank
HERE=Path(__file__).parent;ROOT=Path('/workspace/work/admin-t66-d5-claim');sys.path.insert(0,str(ROOT))
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit,outcome
from m16.approaches.small_ladder_review.portable_cost.t5.geometry import determinant
SOURCE=ROOT/'m16/approaches/exact_ladder/cones.py'
spec=importlib.util.spec_from_file_location('d5_reference_cones',SOURCE);E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

def reference(model):
    h=len(model[0]);ops=E.Ops();audit=Audit();counts=[0]*13;stage=[0]*5;headers=[];rays=[];bits=[];maps=[];good=[]
    original_clip,original_rank=E.clip,E._rank
    def clip(*args,**kwargs):counts[10]+=1;return original_clip(*args,**kwargs)
    def rankfn(*args,**kwargs):counts[9]+=1;return original_rank(*args,**kwargs)
    E.clip,E._rank=clip,rankfn
    def record(c,kind):headers.append((kind,len(c[0])));rays.extend(c[0]);bits.extend(c[1])
    try:
        for cell in E.iter_cells(model,ops,stage):
            record(cell,0);maps.append(cell[2]);gp,bp=outcome(E,cell,model,ops,audit);counts[5]+=len(gp);counts[6]+=len(bp)
            for R,Z in gp:
                record((R,Z),1);pieces=[(R,Z)]
                for k in range(5):
                    new=[]
                    for r,z in pieces:new.extend(v for v in E.clip(r,z,np.eye(5)[k],90+k,ops) if v is not None and len(v[0])>=5)
                    pieces=new
                counts[11]+=len(pieces)
                for r,z in pieces:
                    r=r/np.sum(abs(r),axis=1)[:,None];record((r,z),3)
                    for tr in E.triangulate(r,z,ops):
                        v=r[list(tr)]
                        if determinant(v.T,audit)!=0:good.append(v);counts[7]+=1
                        else:counts[12]+=1
            for c in bp:record(c,2)
    finally:E.clip,E._rank=original_clip,original_rank
    counts[:5]=stage
    return dict(good=np.array(good).reshape(-1,5,5),bad=np.empty((0,5,5)),rays=np.array(rays).reshape(-1,5),bits=np.array(bits,dtype=np.uint64).reshape(-1,2),headers=np.array(headers,dtype=np.int32).reshape(-1,2),maps=np.array(maps).reshape(-1,h,5),counts=np.array(counts),partial_ops=ops.n)

def compare(old,new,model):
    for key in ('headers','bits','counts'):assert np.array_equal(old[key],new[key]),(key,old[key].shape,new[key].shape,old['counts'],new['counts'])
    diff={}
    for key in ('good','bad','rays','maps'):
        assert old[key].shape==new[key].shape,key
        diff[key]=float(np.max(abs(old[key]-new[key]),initial=0))
        if key!='maps':assert diff[key]<=1e-12,(key,diff[key])
    h=len(model[0]);gamma=2*h*np.finfo(float).eps/(1-2*h*np.finfo(float).eps);M=np.zeros((h,5));M[:,0]=abs(model[0])
    for step in (1,2,3,4):M=abs(model[1])@M;M[:,step]=abs(model[0])
    bound=2*np.expm1(4*np.log1p(gamma))*M*(1+gamma)**4
    delta=abs(old['maps']-new['maps']);assert np.all(delta<=bound[None]),'ordinary-arithmetic same-gate map screen'
    diff['map_bound_ratio']=float(np.max(np.divide(delta,bound[None],out=np.zeros_like(delta),where=bound[None]>0),initial=0));return diff

def controls():
    wi=np.array([1.,-1.]);wh=np.array([[.5,-.25],[.125,.75]]);wo=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[.5,.5]])
    yield 'base',(wi,wh,wo)
    yield 'zero_output',(wi,wh,np.zeros((5,2)))
    yield 'duplicate_output',(wi,wh,np.tile(wo[0],(5,1)))
    yield 'no_recurrence',(wi,np.zeros((2,2)),wo)
    for scale in (1e-6,1e6):yield f'coordinate_scale_{scale}',(wi*scale,wh,wo/scale)

def load(path):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True);vals=[state[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')]
    return (vals[0][:,0],vals[1],vals[2]),hashlib.sha256(path.read_bytes()).hexdigest()

def run(trained):
    start=time.process_time();out={'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,HERE/'dd_geometry.cpp',HERE.parent/'dd_clip.cpp',ROOT/'m16/approaches/small_ladder_review/portable_cost/accounting.py',ROOT/'m16/approaches/small_ladder_review/portable_cost/t5/geometry.py')},'cases':[]}
    if not trained:
        out['ranks']=[]
        for value in (0.,.5e-9,1e-9,np.nextafter(1e-9,np.inf),2e-9,1.):
            matrix=np.diag([1.,value,0.,0.,0.]);a=int(np.linalg.matrix_rank(matrix,tol=1e-9));b=rank(matrix);assert a==b;out['ranks'].append([value,a,b])
        cases=list(controls())
    else:
        base,sha=load(Path('/workspace/clean-neurons/work/checkpoints/H2_T5_seed0.pth'));out['checkpoint_sha256']=sha;cases=[]
        for sigma in (0.,.0025,.02):
            rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+rng.normal(size=w.shape)*np.sqrt(np.mean(w*w))*sigma for w in base);cases.append((f'sigma{sigma}',model))
    for name,model in cases:
        row={'name':name,'model_sha256':hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest()}
        for repeat in ('cold','warm'):
            tick=time.process_time();wall=time.perf_counter();old=reference(model);row[f'python_{repeat}_cpu']=time.process_time()-tick;row[f'python_{repeat}_wall']=time.perf_counter()-wall
            tick=time.process_time();wall=time.perf_counter();new=geometry(model);row[f'native_{repeat}_cpu']=time.process_time()-tick;row[f'native_{repeat}_wall']=time.perf_counter()-wall;row['differences']=compare(old,new,model)
        row.update(counts=new['counts'].tolist(),native_components=new['timing'],inherited_partial_ops=old['partial_ops'],warm_cpu_speed=row['python_warm_cpu']/row['native_warm_cpu'] if row['native_warm_cpu']>0 else None,warm_wall_speed=row['python_warm_wall']/row['native_warm_wall']);out['cases'].append(row)
        out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);(HERE/('trained.json' if trained else 'controls.json')).write_text(json.dumps(out,indent=2)+'\n');print(name,row['counts'],row['differences'],row['warm_cpu_speed'],flush=True)
        if out['cpu_seconds']>95:raise RuntimeError('bounded phase cap; partial retained')
    if not trained:
        try:geometry((np.zeros(2),np.eye(2),np.zeros((5,2))))
        except RuntimeError as exc:out['zero_input_rejected']=str(exc)
        else:raise AssertionError('unsupported zero input should fail')
        (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trained',action='store_true');a=p.parse_args()
    try:run(a.trained)
    except Exception as exc:
        (HERE/f'failure_{time.time_ns()}.json').write_text(json.dumps({'trained':a.trained,'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise

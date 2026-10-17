"""Frozen T4 lifecycle parity controls and geometry-only timing."""
import argparse,hashlib,importlib.util,json,resource,time,traceback
from pathlib import Path
import numpy as np
from geometry_native import geometry,rank
HERE=Path(__file__).parent
SOURCE=Path('/workspace/work/admin-t66-d4-claim/m16/approaches/exact_ladder/cones.py')
spec=importlib.util.spec_from_file_location('reference_cones',SOURCE);E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

def reference(model):
    h=len(model[0]);ops=E.Ops();counts=[0]*4;headers=[];rays=[];bits=[];maps=[];good=[];bad=[];npieces=[0,0]
    def record(c,kind):headers.append((kind,len(c[0])));rays.extend(c[0]);bits.extend(c[1])
    for cell in E.iter_cells(model,ops,counts):
        record(cell,0);maps.append(cell[2]);gp,bp=E.outcome_pieces(cell,model,ops,49)
        for kind,pieces,storage in ((1,gp,good),(2,bp,bad)):
            npieces[kind-1]+=len(pieces)
            for R,Z in pieces:
                record((R,Z),kind);storage.extend(R[list(tr)] for tr in E.triangulate(R,Z,ops))
    return dict(good=np.array(good).reshape(-1,4,4),bad=np.array(bad).reshape(-1,4,4),rays=np.array(rays).reshape(-1,4),bits=np.array(bits,dtype=np.uint64).reshape(-1,2),headers=np.array(headers,dtype=np.int32).reshape(-1,2),maps=np.array(maps).reshape(-1,h,4),counts=np.array(counts+npieces+[len(good),len(bad)]),partial_ops=ops.n)

def compare(old,new,model):
    for key in ('headers','bits'):assert np.array_equal(old[key],new[key]),(key,old[key].shape,new[key].shape)
    assert np.array_equal(old['counts'],new['counts'][:8]),('counts',old['counts'],new['counts'])
    diff={}
    for key in ('good','bad','rays','maps'):
        assert old[key].shape==new[key].shape,key
        diff[key]=float(np.max(abs(old[key]-new[key]),initial=0))
        if key!='maps':assert diff[key]<=1e-12,(key,diff[key])
    h=len(model[0]);gamma=2*h*np.finfo(float).eps/(1-2*h*np.finfo(float).eps)
    M=np.zeros((h,4));M[:,0]=abs(model[0])
    for step in (1,2,3):M=abs(model[1])@M;M[:,step]=abs(model[0])
    bound=2*np.expm1(3*np.log1p(gamma))*M*(1+gamma)**3
    delta=abs(old['maps']-new['maps']);assert np.all(delta<=bound[None]),'map accumulation bound'
    diff['map_bound_ratio']=float(np.max(np.divide(delta,bound[None],out=np.zeros_like(delta),where=bound[None]>0),initial=0))
    return diff

def controls():
    wi=np.array([1.,-1.]);wh=np.array([[.5,-.25],[.125,.75]]);wo=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    yield 'base',(wi,wh,wo)
    yield 'zero_output',(wi,wh,np.zeros((4,2)))
    yield 'duplicate_output',(wi,wh,np.tile(wo[0],(4,1)))
    yield 'no_recurrence',(wi,np.zeros((2,2)),wo)
    for scale in (1e-6,1e6):yield f'coordinate_scale_{scale}',(wi*scale,wh,wo/scale)

def load(path):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True)
    vals=[state[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')]
    return (vals[0][:,0],vals[1],vals[2]),hashlib.sha256(path.read_bytes()).hexdigest()

def run(hidden):
    start=time.process_time();out={'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,HERE/'dd_geometry.cpp',HERE.parent/'dd_clip.cpp')},'cases':[]}
    if hidden==0:
        out['ranks']=[]
        for value in (0.,.5e-9,1e-9,np.nextafter(1e-9,np.inf),2e-9,1.):
            matrix=np.diag([1.,value,0.,0.]);a=int(np.linalg.matrix_rank(matrix,tol=1e-9));b=rank(matrix);assert a==b
            out['ranks'].append([value,a,b])
        cases=list(controls())
    else:
        base,sha=load(Path(f'/workspace/clean-neurons/work/checkpoints/H{hidden}_T4_seed0.pth'));out['checkpoint_sha256']=sha;cases=[]
        for sigma in (0.,.0025,.02):
            rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
            cases.append((f'H{hidden}_sigma{sigma}',model))
    for name,model in cases:
        row={'name':name,'model_sha256':hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest()}
        for repeat in ('cold','warm'):
            tick=time.process_time();wall=time.perf_counter();old=reference(model);row[f'python_{repeat}_cpu']=time.process_time()-tick;row[f'python_{repeat}_wall']=time.perf_counter()-wall
            tick=time.process_time();wall=time.perf_counter();new=geometry(model);row[f'native_{repeat}_cpu']=time.process_time()-tick;row[f'native_{repeat}_wall']=time.perf_counter()-wall
            row['differences']=compare(old,new,model)
        row['counts']=new['counts'].tolist();row['native_components']=new['timing'];row['inherited_partial_ops']=old['partial_ops'];row['warm_cpu_speed']=row['python_warm_cpu']/row['native_warm_cpu'] if row['native_warm_cpu']>0 else None;row['warm_wall_speed']=row['python_warm_wall']/row['native_warm_wall'] if row['native_warm_wall']>0 else None;out['cases'].append(row)
        out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (HERE/f'results_h{hidden}.json').write_text(json.dumps(out,indent=2)+'\n');print(name,row['counts'],row['differences'],row['warm_cpu_speed'],flush=True)
        if out['cpu_seconds']>95:raise RuntimeError('bounded cycle cap; partial retained')
    if hidden==0:
        try:geometry((np.zeros(2),np.eye(2),np.zeros((4,2))))
        except RuntimeError as exc:out['unsupported_zero_input']=str(exc)
        else:raise AssertionError('zero-input rejection expected')
        (HERE/f'results_h{hidden}.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hidden',type=int,choices=(0,2,4),default=0);a=p.parse_args()
    try:run(a.hidden)
    except Exception as exc:
        path=HERE/f'failure_h{a.hidden}_{time.time_ns()}.json';path.write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise

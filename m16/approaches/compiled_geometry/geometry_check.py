"""Frozen full native T3 geometry controls and three-case paired workload."""
import hashlib,json,resource,time
import numpy as np
from dd_profile import E,ladder,load,HERE,ROOT
from controls import fixtures
from geometry_native import geometry,rank

def reference(model):
    ops=E.Ops();counts=[0]*3;headers=[];rays=[];bits=[];maps=[];good=[];bad=[];npieces=[0,0]
    def record(c,kind):headers.append((kind,len(c[0])));rays.extend(c[0]);bits.extend(c[1])
    for cell in E.iter_cells(model,ops,counts):
        record(cell,0);maps.append(cell[2]);gp,bp=E.outcome_pieces(cell,model,ops,33)
        for kind,pieces,storage in ((1,gp,good),(2,bp,bad)):
            npieces[kind-1]+=len(pieces)
            for R,Z in pieces:
                record((R,Z),kind)
                storage.extend(R[list(tr)] for tr in E.triangulate(R,Z,ops))
    return dict(good=np.array(good).reshape(-1,3,3),bad=np.array(bad).reshape(-1,3,3),rays=np.array(rays),bits=np.array(bits,dtype=np.uint64),headers=np.array(headers,dtype=np.int32),maps=np.array(maps),counts=np.array(counts+npieces+[len(good),len(bad)]))

def compare(old,new,model):
    for key in ('headers','bits'):assert np.array_equal(old[key],new[key]),key
    assert np.array_equal(old['counts'],new['counts'][:7]),'counts'
    differences={}
    for key in ('good','bad','rays','maps'):
        assert old[key].shape==new[key].shape,key
        differences[key]=float(np.max(abs(old[key]-new[key]),initial=0))
        if key!='maps':assert differences[key]<=1e-12,(key,differences[key])
    h=len(model[0]);gamma=2*h*np.finfo(float).eps/(1-2*h*np.finfo(float).eps)
    M=np.zeros((h,3));M[:,0]=abs(model[0])
    for step in (1,2):M=abs(model[1])@M;M[:,step]=abs(model[0])
    bound=2*np.expm1(2*np.log1p(gamma))*M*(1+gamma)**2
    delta=abs(old['maps']-new['maps']);assert np.all(delta<=bound[None]),'weight-derived map accumulation bound'
    differences['map_roundoff_bound_ratio']=float(np.max(np.divide(delta,bound[None],out=np.zeros_like(delta),where=bound[None]>0),initial=0))
    differences['maps_relative_to_max']=differences['maps']/max(1.,float(np.max(abs(old['maps']),initial=0)))
    return differences

def measure(result):
    sums=[]
    for key in ('good','bad'):
        v=result[key];sums.append(float(E.cone_measure(v,E.Ops())[0].sum()) if len(v) else 0.)
    return 1-sums[1],sum(sums)

def run():
    start=time.process_time();out={'controls':[],'ranks':[],'cases':[]}
    for value in (0.,.5e-9,1e-9,np.nextafter(1e-9,np.inf),2e-9,1.):
        matrix=np.diag([1.,value,0.]);a=int(np.linalg.matrix_rank(matrix,tol=1e-9));b=rank(matrix);assert a==b
        out['ranks'].append(dict(second_singular=value,numpy_rank=a,native_rank=b))
    for name,model in fixtures():
        if len(model[2])!=3:continue
        old=reference(model);new=geometry(model);differences=compare(old,new,model);out['controls'].append(dict(name=name,differences=differences,counts=new['counts'].tolist(),value=measure(new)))
    base,sha=load(ROOT/'work/checkpoints/H16_T3_seed0.pth');out['checkpoint_sha256']=sha
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        old=reference(model);new=geometry(model);diff=compare(old,new,model);row={'sigma':sigma,'geometry_differences':diff,'native_counts':new['counts'].tolist()}
        for repeat in ('cold','warm'):
            wall=time.perf_counter();tick=time.process_time();value,info=ladder.exact_acc(model,check=True);row[f'python_{repeat}_cpu']=time.process_time()-tick;row[f'python_{repeat}_wall']=time.perf_counter()-wall
            wall=time.perf_counter();tick=time.process_time();native=geometry(model);nv,partition=measure(native);row[f'native_{repeat}_cpu']=time.process_time()-tick;row[f'native_{repeat}_wall']=time.perf_counter()-wall
            assert abs(nv-value)<1e-12;row[f'native_{repeat}_components']=native['timing']
        row.update(python_value=value,native_value=nv,value_difference=nv-value,partition_mass=partition,warm_cpu_speed=row['python_warm_cpu']/row['native_warm_cpu'],warm_wall_speed=row['python_warm_wall']/row['native_warm_wall'])
        out['cases'].append(row);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256((HERE/'dd_geometry.cpp').read_bytes()).hexdigest(),clip_sha256=hashlib.sha256((HERE/'dd_clip.cpp').read_bytes()).hexdigest())
        (HERE/'geometry_results.json').write_text(json.dumps(out,indent=2)+'\n');print(sigma,'speed',row['warm_cpu_speed'],row['warm_wall_speed'],'diff',row['value_difference'],'geometry',diff,flush=True)
if __name__=='__main__':
    try:run()
    except Exception as exc:
        import traceback
        (HERE/'geometry_failure.json').write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise

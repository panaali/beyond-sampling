"""Independent fixed trained-corpus replay, with cold-cache and failure telemetry."""
import time
START=time.process_time()
import argparse,hashlib,json,resource
from pathlib import Path
import numpy as np
from . import author_snapshot as M
HERE=Path(__file__).parent
SOURCE_HASH='6617ddd0a042d4af5fbc899e0542b9e31c1ee23f5ab8d47e5fb8ee2c005f05a6'
CAPTURES={'quadratic':'07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581','nearest':'8c291005cafccb047e6a0af2441d76505ce6caf8f9e56b1641d437079842f52a'}
assert hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()==SOURCE_HASH
IMPORT_CPU=time.process_time()-START

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=list(CAPTURES));a=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(105,107));start=time.process_time()
    path=HERE.parent/'class_region_t4'/(a.kind+'_exact_mass_calls.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==CAPTURES[a.kind]
    data=np.load(path);rays=data['rays'];old=data['values'];source=json.loads((path.parent/(a.kind+'_exact.json')).read_text());load_cpu=time.process_time()-start
    M._GL.clear();original_gl=M._gl;events=[]
    def counted_gl(n):
        miss=n not in M._GL;t=time.process_time();result=original_gl(n)
        if miss:events.append(dict(order=int(n),cpu=time.process_time()-t))
        return result
    M._gl=counted_gl;values=[];costs=[];failures=[];out=dict(kind=a.kind,source_sha256=SOURCE_HASH,capture_sha256=CAPTURES[a.kind],complete=False,planned_calls=len(rays),import_cpu=IMPORT_CPU,capture_load_cpu=load_cpu,cold_cache=True,full_portable_ops_certified=False)
    def save():
        valid=[v for v in values if np.isfinite(v)];out.update(completed_calls=len(values),failures=failures,partial_finite_sum=float(sum(valid)),partial_author_ops=int(sum(costs)),gl_setup_events=events,cpu=time.process_time()-start)
        (HERE/(a.kind+'_results.json')).write_text(json.dumps(out,indent=2)+'\n')
    try:
        for i,R in enumerate(rays):
            if time.process_time()-start>100:raise TimeoutError('Frozen100CPU cap')
            ops=M.Ops()
            try:
                value=float(M.mass4_rays(R,ops))
                if not np.isfinite(value):raise ArithmeticError('Nonfinite mass')
            except Exception as exc:
                failures.append(dict(index=i,type=type(exc).__name__,message=str(exc),partial_author_ops=ops.n));value=float('nan')
            values.append(value);costs.append(ops.n)
            if (i+1)%512==0:save()
        out['complete']=True
    except Exception as exc:out.update(cycle_failure=type(exc).__name__+': '+str(exc))
    finally:M._gl=original_gl
    v=np.array(values);c=np.array(costs,dtype=np.int64)
    np.savez_compressed(HERE/(a.kind+'_values.npz'),values=v,author_ops=c)
    if out['complete'] and not failures:
        estimate=1-float(v.sum());families=[];offset=0
        for row in source['rows']:
            n=row['stats']['simplices'];part=float(v[offset:offset+n].sum());families.append(dict(index=row['index'],truth=row['truth'],sigma=row['sigma'],mass_calls=n,old_error=row['error'],new_error=part));offset+=n
        assert offset==len(v)
        out.update(estimate=estimate,reference=.477013125994119,signed_reference_error=estimate-.477013125994119,old_estimate=source['estimate'],negative_count=int(sum(v<0)),negative_sum=float(v[v<0].sum()),minimum_mass=float(v.min()),zero_count=int(sum(v==0)),mass_calls_author_ops=int(c.sum()),mean_author_ops=float(c.mean()),max_author_ops=int(c.max()),families=families)
    save();print(json.dumps({k:v for k,v in out.items() if k not in ('families','failures')},indent=2));print('failures',len(failures),flush=True)
if __name__=='__main__':main()

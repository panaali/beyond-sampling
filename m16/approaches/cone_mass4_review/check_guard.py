"""Local review-only positive fallback; author source and untouched calls preserved."""
import json,time,resource,hashlib
from pathlib import Path
import numpy as np
from . import author_snapshot as M
from .check_controls import positive_mass
from .replay import CAPTURES,SOURCE_HASH
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time();out=dict(source_sha256=SOURCE_HASH,source_modified=False,corpora=[],portable_cost_certified=False)
    for kind in ('quadratic','nearest'):
        begin=time.process_time();path=HERE.parent/'class_region_t4'/(kind+'_exact_mass_calls.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==CAPTURES[kind]
        rays=np.load(path)['rays'];M._GL.clear();vals=[];fallbacks=[];unresolved=[];ops_total=0;bound_total=0.;events=[]
        original_gl=M._gl
        def gl(n):
            miss=n not in M._GL;t=time.process_time();value=original_gl(n)
            if miss:events.append(dict(order=n,cpu=time.process_time()-t))
            return value
        M._gl=gl
        try:
            for i,R in enumerate(rays):
                op=M.Ops();failure=None
                try:
                    val=float(M.mass4_rays(R,op))
                    if not np.isfinite(val):raise ArithmeticError('Nonfinite author value')
                except Exception as exc:val=float('nan');failure=type(exc).__name__+': '+str(exc)
                ops_total+=op.n
                if failure or val<0:
                    same=bool(np.all((R.min(0)>=0)|(R.max(0)<=0)))
                    if same:
                        t=time.process_time();old=val;val,bnd=positive_mass(abs(R),20);bound_total+=bnd
                        fallbacks.append(dict(index=i,original_value=old if np.isfinite(old) else None,original_failure=failure,value=val,integration_only_bound=bnd,nodes=8000,integrand_arithmetic=39*8000,cpu=time.process_time()-t))
                    else:unresolved.append(dict(index=i,value=val if np.isfinite(val) else None,failure=failure,reason='mixed-sign original rays outside fixed positive fallback'))
                vals.append(val)
        finally:M._gl=original_gl
        v=np.array(vals);row=dict(kind=kind,count=len(v),author_ops=ops_total,fallbacks=fallbacks,unresolved=unresolved,negative_count=int(sum(v<0)),fallback_integration_only_bound=bound_total,cold_gl_events=events,cpu=time.process_time()-begin,full_accuracy_certified=False)
        if np.isfinite(v).all():row.update(estimate=1-float(v.sum()),signed_reference_error=1-float(v.sum())-.477013125994119)
        else:row['aggregate_accuracy_not_reported']=True
        np.savez_compressed(HERE/(kind+'_guard_values.npz'),values=v);out['corpora'].append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('cold_gl_events',)},indent=2),flush=True)
    out.update(complete=True,cpu=time.process_time()-start)
    (HERE/'guard_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

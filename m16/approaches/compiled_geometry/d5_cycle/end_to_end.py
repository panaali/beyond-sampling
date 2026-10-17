"""Unchanged-rule paired numerical cold/warm geometry+mass workload."""
import hashlib,json,resource,time,traceback
from pathlib import Path
import numpy as np
from check import load
from adapter import calculate,prior,T71
HERE=Path(__file__).parent

def run():
    start=time.process_time();base,sha=load(Path('/workspace/clean-neurons/work/checkpoints/H2_T5_seed0.pth'));out={'checkpoint_sha256':sha,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (T71/'counted.py',T71/'fixed_tangent.py',T71/'primitive.py',T71/'estimator.py',HERE/'dd_geometry.cpp',HERE/'adapter.py')},'cases':[]}
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+rng.normal(size=w.shape)*np.sqrt(np.mean(w*w))*sigma for w in base)
        row={'sigma':sigma,'model_sha256':hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest(),'pairs':[]};out['cases'].append(row)
        for phase in ('cold','warm'):
            tick=time.process_time();wall=time.perf_counter();old,oldinfo=prior.calculate(model);oldcpu=time.process_time()-tick;oldwall=time.perf_counter()-wall
            tick=time.process_time();wall=time.perf_counter();new,newinfo=calculate(model);newcpu=time.process_time()-tick;newwall=time.perf_counter()-wall
            pair=dict(phase=phase,python_estimate=old,native_estimate=new,difference=new-old,python_cpu=oldcpu,python_wall=oldwall,native_cpu=newcpu,native_wall=newwall,cpu_ratio=oldcpu/newcpu if newcpu>0 else None,wall_ratio=oldwall/newwall,python=oldinfo,native=newinfo)
            row['pairs'].append(pair);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);(HERE/'end_to_end_results.json').write_text(json.dumps(out,indent=2)+'\n')
            assert oldinfo['correct_simplices']==newinfo['correct_simplices'];assert abs(new-old)<=1e-12,(sigma,phase,new-old)
            print(sigma,phase,new,new-old,'CPU ratio',pair['cpu_ratio'],flush=True)
            if out['cpu_seconds']>95:raise RuntimeError('bounded phase cap; partial retained')
if __name__=='__main__':
    try:run()
    except Exception as exc:
        (HERE/f'end_to_end_failure_{time.time_ns()}.json').write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise

"""Predeclared two-plane exact Fraction topology/control, not a reference sweep."""
import json,time,resource
from pathlib import Path
import numpy as np
from check import HERE,new,load,exact_circle

def run():
    start=time.process_time();model,sha=load(Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    g=np.random.default_rng(660001).standard_normal((32,10,2));q,r=np.linalg.qr(g)
    q*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,None,:];out=dict(checkpoint_sha256=sha,cases=[])
    for index in (0,31):
        u,v=q[index,:,0],q[index,:,1];tick=time.process_time();truth,oracle=exact_circle(model,u,v)
        elapsed=time.process_time()-tick;row=dict(plane=index,exact_probability=truth,oracle=oracle,oracle_cpu=elapsed,native={})
        for engine in ('factored_axis','factored_mixed_axis'):
            val,stats=new.walk(model,u[None],v[None],engine=engine)
            row['native'][engine]=dict(probability=float(val[0]),difference=float(val[0]-truth),stats=stats)
        out['cases'].append(row);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (HERE/'exact_trained_axis_results.json').write_text(json.dumps(out,indent=2)+'\n')
        print(index,truth,oracle,'oracle CPU',elapsed,'native counts',row['native']['factored_axis']['stats']['gate_pieces'],flush=True)
        if index==0 and elapsed>=10:
            out['skipped_plane31']='predeclared first-oracle CPU threshold';break
    out['cpu_seconds']=time.process_time()-start
    (HERE/'exact_trained_axis_results.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':run()

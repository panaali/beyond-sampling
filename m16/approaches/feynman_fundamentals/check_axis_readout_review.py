"""Closed-form axis readout controls including a positive-volume output tie."""
import importlib.util,json,time,hashlib
from pathlib import Path
import numpy as np
start=time.process_time()
base=Path('/workspace/work/recover-small_h2_78/m16/approaches/reset_root_factor')
spec=importlib.util.spec_from_file_location('axis_review_native',base/'native.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
u=np.array([0.,1.,0.,0.]);v=np.array([0.,0.,0.,1.]);rows=[]
# The late-reset circle is the same exact fixture as check_late_reset_review.
# If the positive ray predicts3, accuracy3/8. If rows2 and3 tie at the top,
# first-index winner2 gives the whole negative chart correct and the positive
# chart incorrect, hence1/2. Dead final hidden state always predicts0.
for tied,expected in ((False,.375),(True,.5)):
    wo=np.zeros((4,3));wo[:,0]=[0.,-1.,1. if tied else -2.,1.]
    for scale in (1.,1e-12,1e12):
        model=(np.array([.7,0.,0.])*scale,-np.eye(3),wo)
        for frame,(a,b) in enumerate(((u,v),(v,-u),(-u,-v),(-v,u))):
            for engine in ('factored_axis','factored_mixed_axis'):
                values,stats=native.walk(model,a[None],b[None],engine=engine)
                assert abs(values[0]-expected)<1e-14
                assert abs(stats['coverage'][0]-1)<1e-14
                rows.append(dict(tied=tied,scale=scale,frame=frame,engine=engine,
                                 expected=expected,value=float(values[0]),stats=stats))
out=dict(rows=rows,max_error=max(abs(r['value']-r['expected']) for r in rows),
         cpu_seconds=time.process_time()-start,
         source_hashes={f:hashlib.sha256((base/f).read_bytes()).hexdigest()
                        for f in ('factored_axis.cpp','factored_mixed_axis.cpp','native.py')})
Path(__file__).with_name('axis_readout_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

"""Closed-form circle control independent of the author's Fraction oracle."""
import importlib.util
import hashlib
import json
import time
from pathlib import Path
import numpy as np

START=time.process_time()
base=Path('/workspace/work/recover-small_h2_78/m16/approaches/reset_root_factor')
spec=importlib.util.spec_from_file_location('independent_reset_native',base/'native.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
# x=(0, sigma, 0, sigma*t). With Wh=-I and positive Wi, stage2
# is structurally dead regardless of stage1. Final prediction3 iff x3>0,
# else0. Stable-index input ties give correctness t in(0,1) on sigma+
# and t in(0,infinity) on sigma-. Mass=(pi/4+pi/2)/(2pi)=3/8.
u=np.array([0.,1.,0.,0.]);v=np.array([0.,0.,0.,1.])
wo=np.array([[0.,0.,0.],[-1.,-1.,0.],[-2.,-2.,0.],[1.,1.,0.]])
rows=[]
for scale in (1.,1e-12,1e12):
    model=(np.array([.7,1.3,0.])*scale,-np.eye(3),wo)
    for frame,(a,b) in enumerate(((u,v),(v,-u),(-u,-v),(-v,u))):
        for engine in ('factored','factored_mixed'):
            values,stats=native.walk(model,a[None],b[None],engine=engine)
            assert abs(values[0]-.375)<1e-14
            assert abs(stats['coverage'][0]-1)<1e-14
            assert stats['late_zero_steps'][0]>0
            rows.append(dict(scale=scale,frame=frame,engine=engine,value=float(values[0]),stats=stats))
out=dict(expected=.375,rows=rows,max_error=max(abs(r['value']-.375) for r in rows),
         cpu_seconds=time.process_time()-START,
         source_hashes={f:hashlib.sha256((base/f).read_bytes()).hexdigest()
                        for f in ('factored.cpp','factored_mixed.cpp','native.py')},
         scope='24 closed-form late-reset/constant-input/zero-Wi/tie/phase/scale checks; no generic topology certificate.')
Path(__file__).with_name('late_reset_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

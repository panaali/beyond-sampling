"""Memoryless analytic control for the old operator's measure and mixture."""
from collections import defaultdict
from pathlib import Path
import json,time
import numpy as np
from terminal import kernel,MiddleTarget
from interpolation import InterpolatedMiddle
start=time.process_time();wi=np.array([.8,-.5]);wh=np.zeros((2,2))
wo=np.array([[-.1,.8],[-.3,-.1],[1.,-.2],[.2,.1]])
model=wi,wh,wo;states=np.maximum(np.array([-1.,0.,1.])[:,None]*wi,0.)
old=InterpolatedMiddle(model,states);error=0.;rows=[]
for u in (-1.,0.,1.):
    values,bound=old.values(u);stats=defaultdict(float)
    L=kernel.mass(-6.,u,stats);U=kernel.mass(u,6.,stats)
    low=kernel.mass(0.,u,stats)/L;high=kernel.mass(max(u,0.),6.,stats)/U
    expected=np.array([low,(low+high)/2])
    error=max(error,float(np.max(abs(values-expected))))
    for state in states:
        pair,_=MiddleTarget(model,state,u).probabilities()
        error=max(error,float(np.max(abs(pair-expected))))
    rows.append(dict(u=u,values=values.tolist(),analytic=expected.tolist(),kernel_error_diagnostic=bound))
assert error<1e-10
out=dict(max_error=error,rows=rows,domain=old.domain,cpu_seconds=time.process_time()-start)
(Path(__file__).resolve().parent/'interpolation_controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

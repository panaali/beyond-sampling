"""Direct scalar algorithm comparison, including boundary-only ties."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.special import ndtr
from terminal_cache import TerminalCache

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from engine import terminal
sys.path.pop(0)
wi=np.array([-.8,.5]);wh=np.array([[.2,.3],[-.7,.1]])
wo=np.array([[0.,0.],[-1.,-1.],[-2.,-2.],[1.,1.]])
states=np.array([[0.,0.],[.2,0.],[0.,.3],[.4,.6]])
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
trained=(w['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
         w['rnn.weight_hh_l0'].numpy().astype(float),w['linear.weight'].numpy().astype(float))
permuted=(trained[0],trained[1],trained[2][[0,1,3,2]])
cases=[('constant',(wi,wh,np.zeros((4,2)))),
       ('duplicate',(wi,wh,np.tile([.3,-.7],(4,1)))),
       ('Wh0',(wi,np.zeros((2,2)),wo)),('original',trained),('permuted',permuted)]
start=time.process_time();out=dict(rows=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
for name,model in cases:
    cache=TerminalCache(model,states)
    source_stats=defaultdict(float)
    queries=np.unique(np.r_[-4.,-1.,-.1,0.,.1,1.,4.,cache.knots])
    row=dict(name=name,queries=[],cache_work=None)
    out['rows'].append(row)
    for u in queries:
        q=float((ndtr(u)-ndtr(-6))/(1-2*ndtr(-6)))
        value=cache.values(float(u),q)
        exact=np.array([terminal(model,h,float(u),q,3,6.,source_stats) for h in states])
        error=float(np.max(abs(value-exact)))
        row['queries'].append(dict(u=float(u),error=error))
        row['cache_work']=dict(cache.stats);row['direct_work']=dict(source_stats)
        out['cpu_seconds']=time.process_time()-start
        (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
        if error>2e-10:
            row['failure']=dict(error=error,cached=value.tolist(),direct=exact.tolist())
            (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
            raise AssertionError((name,u,error))
        if name in ('constant','duplicate'):assert value.max()==0.
        if name=='Wh0':assert np.max(abs(value-(0. if u==0 else 1.)))<1e-10
    scaled=(model[0]*1e-16,model[1],model[2])
    small=TerminalCache(scaled,states*1e-16)
    scale_error=max(float(np.max(abs(cache.values(u)-small.values(u))))
                    for u in (-4.,-1.,-.1,0.,.1,1.,4.))
    row['scale_error']=scale_error;assert scale_error<2e-10
out['cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(maximum_error=max(q['error'] for r in out['rows'] for q in r['queries']),
                     maximum_scale_error=max(r['scale_error'] for r in out['rows']),
                     cpu_seconds=out['cpu_seconds'])))

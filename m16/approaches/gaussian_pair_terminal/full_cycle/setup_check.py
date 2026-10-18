import json,time
from pathlib import Path
import numpy as np
import torch
from driver import setup
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=a[0].ravel(),a[1],a[2]
start=time.process_time();out={}
try:
 data=setup(model);out.update(panels=len(data['points'])-1,predicted_pair_queries=data['pair_query_budget'],labels=data['labels'],pair_objects=sum(map(len,data['pairs'].values())),points=data['points'].tolist(),work=dict(data['stats']))
except Exception as exc:out['failure']=repr(exc)
out['cpu_seconds']=time.process_time()-start
(HERE/'setup.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('points','work')}))
if 'failure' in out:raise SystemExit(1)

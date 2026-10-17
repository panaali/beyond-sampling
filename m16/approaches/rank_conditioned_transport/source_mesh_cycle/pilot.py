"""Frozen one-model, matched-node stage1 mesh comparison."""
import hashlib,json
from pathlib import Path
import time
import torch
from driver import estimate
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(reference=.30773074302392744,rows=[],
    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [HERE/'mesh.py',HERE/'driver.py',HERE.parent/'engine.py',HERE.parent/'polynomial.py',HERE.parent/'breakpoints.py',HERE.parent/'panel.py']})
for order in (2,4):
    for mode in ('baseline','physical','reversed'):
        row=dict(order=order,mode=mode)
        try:
            value,stats=estimate(model,9,order,mode,seconds=100-(time.process_time()-start))
            row.update(estimate=value,error=value-out['reference'],stats=stats)
        except Exception as exc:row['failure']=repr(exc)
        out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
        (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
        if 'failure' in row:raise SystemExit(1)
summary={}
for mode in ('baseline','physical','reversed'):
    a,b=[r for r in out['rows'] if r['mode']==mode]
    summary[mode]=dict(refinement_change=abs(a['estimate']-b['estimate']),error=b['error'],cpu=b['stats']['cpu_seconds'],error_squared_cpu=b['error']**2*b['stats']['cpu_seconds'])
out['summary']=summary
(HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))

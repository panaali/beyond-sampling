import hashlib,json,time
from pathlib import Path
import torch
from driver import estimate
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(reference=.30773074302392744,rows=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('core.py','driver.py')})
for order in (2,4):
    for direct in (False,True):
        row=dict(order=order,direct=direct)
        try:
            value,stats=estimate(model,9,order,direct,seconds=100-(time.process_time()-start))
            row.update(estimate=value,error=value-out['reference'],stats=stats)
        except Exception as exc:row['failure']=repr(exc)
        out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
        (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
        if 'failure' in row:raise SystemExit(1)
for direct in (False,True):
    a,b=[r for r in out['rows'] if r['direct']==direct]
    out['direct' if direct else 'baseline']=dict(refinement_change=abs(a['estimate']-b['estimate']),error=b['error'],cpu=b['stats']['cpu_seconds'],error_squared_cpu=b['error']**2*b['stats']['cpu_seconds'])
(HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('baseline','direct','total_cpu_seconds')}))

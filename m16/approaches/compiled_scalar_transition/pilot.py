import hashlib,json,time
from pathlib import Path
import torch
from driver import estimate
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(rows=[],build=json.loads((HERE/'build.json').read_text()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('kernel.cpp','native.py','driver.py')})
for native in (False,True):
    row=dict(native=native)
    try:
        value,stats=estimate(model,9,2,native,seconds=100-(time.process_time()-start))
        row.update(estimate=value,stats=stats)
    except Exception as exc:row['failure']=repr(exc)
    out['rows'].append(row);out['cpu_seconds']=time.process_time()-start
    (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    if 'failure' in row:raise SystemExit(1)
a,b=out['rows'];out['difference']=abs(a['estimate']-b['estimate']);assert out['difference']<=3e-12
out['runtime_speed_ratio']=a['stats']['cpu_seconds']/b['stats']['cpu_seconds']
out['first_use_cpu_speed_ratio']=a['stats']['cpu_seconds']/(b['stats']['cpu_seconds']+out['build']['child_cpu_seconds'])
out['same_rule_same_interpolant']=True
(HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('difference','runtime_speed_ratio','first_use_cpu_speed_ratio','cpu_seconds')}))

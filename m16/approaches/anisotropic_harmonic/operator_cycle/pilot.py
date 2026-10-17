"""Cold full-driver equivalence and initial cost gate."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from adapter import estimate,dense_estimate

HERE=Path(__file__).resolve().parent
start=time.process_time()
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
w=torch.load(path,weights_only=True,map_location='cpu')
h2=(w['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
    w['rnn.weight_hh_l0'].numpy().astype(float),w['linear.weight'].numpy().astype(float))
z=np.load('/workspace/clean-neurons/m16/approaches/arc_m43_repro/m43_weights.npz')
h4=tuple(z[k].astype(float) for k in ('Wi','Wh','Wo'));h4=(h4[0].ravel(),h4[1],h4[2])
out=dict(rows=[],controls=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
for name,model in [('H2T3',h2),('H4T3',h4)]:
    old,oldstats=dense_estimate(model)
    value,stats=estimate(model)
    row=dict(name=name,dense=dict(estimate=old,stats=oldstats),
             real_operator=dict(estimate=value,stats=stats),difference=abs(value-old))
    out['rows'].append(row)
    assert row['difference']<1e-9
    for kind,changed in [('scaled',(model[0]*1e-16,model[1],model[2])),
                         ('zero',(model[0],model[1],np.zeros_like(model[2]))),
                         ('duplicate',(model[0],model[1],np.tile(model[2][0],(3,1))))]:
        check,checkstats=estimate(changed)
        target=value if kind=='scaled' else 1/3
        out['controls'].append(dict(name=name,kind=kind,estimate=check,error=abs(check-target),stats=checkstats))
        assert abs(check-target)<1e-9
    out['total_cpu_seconds']=time.process_time()-start
    (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(name=name,difference=row['difference'],dense_cpu=oldstats['cpu_seconds'],
        real_cpu=stats['cpu_seconds'],dense_accounted_ops=oldstats['accounted_scalar_arithmetic_upper'],
        real_accounted_ops=stats['accounted_scalar_arithmetic_upper'])),flush=True)
